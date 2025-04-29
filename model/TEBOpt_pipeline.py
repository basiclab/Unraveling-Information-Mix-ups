"""
Util functions based on Diffuser framework and FreePromptEditing, Syngen, and A&E.
"""


import os
import torch
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
import json

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
from typing import Optional, List, Callable, Dict, Union

from model.attentions_utils import AttentionStore
import torch.distributions as dist

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    attention_maps = attention_store.get_cur_attention()
    out = []
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

class TEBPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        # reference: https://github.com/huggingface/diffusers/blob/b69fd990ad8026f21893499ab396d969b62bb8cc/src/diffusers/schedulers/scheduling_ddim.py#L342
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # self.config.prediction_type == "epsilon"
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def _symmetric_kl(self, attention_map1, attention_map2):
        # Convert map into a single distribution: 16x16 -> 256
        if len(attention_map1.shape) > 1:
            attention_map1 = attention_map1.reshape(-1)
        if len(attention_map2.shape) > 1:
            attention_map2 = attention_map2.reshape(-1)

        p = dist.Categorical(probs=attention_map1)
        q = dist.Categorical(probs=attention_map2)

        kl_divergence_pq = dist.kl_divergence(p, q)
        kl_divergence_qp = dist.kl_divergence(q, p)

        avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
        return avg_kl_divergence

    def _norm_attnmap(self, attention_maps):
        last_idx = -1
        
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        return attention_for_text

    @staticmethod
    def _update_text_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents
    
    def _cal_text_loss(self, gen_text_emb, pure_text_emb_list, indices_to_balance, given_prompt_len, loss_log):
        # gen_text_emb.size() = 1, 77, 768 
        # pure_text_emb.size() = 1, 768 (every tensor inside pure_text_emb_list)

        # sim_pair [[gen_emb, [pure_emb]], [gen_emb, [pure_emb]]]
        sim_pair, dissim_pair = [], []
        for idx, selected_indice in enumerate(indices_to_balance):
            sim_pair.append([selected_indice, [idx]])
            all_neg_list = [i for i in range(1, given_prompt_len-1) if i != selected_indice]    # exclude sot and eot
            dissim_pair.append([selected_indice, all_neg_list])

        sim_loss, dissim_loss = [], []
        
        # calculate sim_loss
        for gen_idx, pure_idx_list in sim_pair:
            # Normalize the embeddings if using cosine similarity
            text_emb_norm = F.normalize(gen_text_emb[:, gen_idx, :], dim=-1)
            for pure_idx in pure_idx_list:
                pure_emb_norm = F.normalize(pure_text_emb_list[pure_idx], dim=-1)
                # Cosine similarity calculations
                cos_sim_target = F.cosine_similarity(text_emb_norm, pure_emb_norm)
                sim_loss.append(cos_sim_target)
                
        # calculate dissimilarity
        for gen_idx, gen_other_idx_list in dissim_pair:
            # Normalize the embeddings if using cosine similarity
            text_emb_norm = F.normalize(gen_text_emb[:, gen_idx, :], dim=-1)
            for gen_other_idx in gen_other_idx_list:
                gen_other_emb_norm = F.normalize(gen_text_emb[:, gen_other_idx, :], dim=-1)
                # Cosine similarity calculations
                cos_sim_negative = F.cosine_similarity(text_emb_norm, gen_other_emb_norm)
                dissim_loss.append(cos_sim_negative)
                
        positive_sim_loss = min(sim_loss)
        negative_sim_loss = sum(dissim_loss) / len(dissim_loss)

        loss = -positive_sim_loss + negative_sim_loss
        loss_log.write("\t loss = {:.4f} \n".format(loss.item()))
        pos_numeric_values = ["{:.4f}".format(t.item()) for t in sim_loss]
        neg_numeric_values = ["{:.4f}".format(t.item()) for t in dissim_loss]
        loss_log.write("\t pos_loss = {} \n".format(pos_numeric_values))
        loss_log.write("\t neg_loss = {}  \n".format(neg_numeric_values))
        
        return loss, positive_sim_loss, negative_sim_loss

    def _cal_textEMB_sim(self, text_embeddings, save_path, prompt, seed, status,
                            attention_store: AttentionStore,
                            indices_to_balance: List[int],
                            attention_res: int = 16,
                            normalize_eot: bool = False):
        
        # calculate text embedding similarity
        obj1_emb_norm = F.normalize(text_embeddings[1, 2, :].unsqueeze(0), dim=-1)
        obj2_emb_norm = F.normalize(text_embeddings[1, 5, :].unsqueeze(0), dim=-1)
        # Cosine similarity calculations
        cos_sim = F.cosine_similarity(obj1_emb_norm, obj2_emb_norm)
        
        # calculate cross-attention map similarity
        attention_maps = aggregate_attention(attention_store=attention_store,
                                            res=attention_res,
                                            from_where=("up", "down", "mid"),
                                            is_cross=True,
                                            select=0)
        
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_balance = [index - 1 for index in indices_to_balance]

        # calculate map distance
        obj1_attn_map = attention_for_text[:, :, indices_to_balance[0]]
        obj2_attn_map = attention_for_text[:, :, indices_to_balance[1]]
        cross_map_dist = self._symmetric_kl(obj1_attn_map, obj2_attn_map)


        new_entry_data = {
                "{}_{}.png".format(prompt, seed): {
                    '{}_text_sim'.format(status): cos_sim.item(),
                    '{}_cross_map_dist'.format(status): cross_map_dist.item(),
                }
            }
        
        # Check if the file exists
        if os.path.exists(os.path.join(save_path, "_raw_similarity.json")):
            # Read the existing data
            with open(os.path.join(save_path, "_raw_similarity.json"), 'r') as file:
                data = json.load(file)
                # Append new data to the existing lists
                data.update(new_entry_data)
        else:
            # No file exists, create new data structure
            data = new_entry_data

        with open(os.path.join(save_path, "_raw_similarity.json"), 'w') as f:
            json.dump(data, f, sort_keys=False, indent=4)

    def _textEMB_step(
            self,
            text_embeddings,
            pure_embedding_list,
            indices_to_balance,
            given_prompt_len,
            step_size,
            max_opt_iter,
            loss_log,
    ):
        with torch.enable_grad():
            text_embeddings_cond = text_embeddings.clone().detach().requires_grad_(True)
            loss, positive_sim_loss, negative_sim_loss = self._cal_text_loss(text_embeddings_cond, pure_embedding_list, indices_to_balance, given_prompt_len, loss_log)

            threshold_pos, threshold_neg = 0.95, 0.25
            intertation = 0
            while -positive_sim_loss+negative_sim_loss > -threshold_pos+threshold_neg:
                intertation+=1
                loss_log.write("\t --- Text embedding optimization ---  \n")
                loss_log.write("\t Iteration = {} | loss = {:.4f}, pos_loss = {:.4f}, neg_loss = {:.4f}  \n".format(intertation, loss.item(), positive_sim_loss.item(), negative_sim_loss.item()))
                # print("Iteration = {} | loss = {:.4f}, pos_loss = {:.4f}, neg_loss = {:.4f}  \n".format(intertation, loss.item(), positive_sim_loss.item(), negative_sim_loss.item()))

                text_embeddings_cond = self._update_text_latent(latents=text_embeddings_cond, loss=loss, step_size=step_size)
                loss, positive_sim_loss, negative_sim_loss = self._cal_text_loss(text_embeddings_cond, pure_embedding_list, indices_to_balance, given_prompt_len, loss_log)

                if intertation > max_opt_iter:
                    loss_log.write("Exceed max opt iter | loss = {:.4f}, pos_loss = {:.4f}, neg_loss = {:.4f}  \n".format(loss.item(), positive_sim_loss.item(), negative_sim_loss.item()))
                    print("Exceed max opt iter | loss = {:.4f}, pos_loss = {:.4f}, neg_loss = {:.4f}  \n".format(loss.item(), positive_sim_loss.item(), negative_sim_loss.item()))
                    break
            text_embeddings = text_embeddings_cond
        return text_embeddings
        
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        attention_store: AttentionStore,
        indices_to_balance: List[int],
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        save_path: Union[str, List[str]] = None,
        seed: Optional[int] = None,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        concat_pure_text_emb: bool = False,
        text_emb_optimize: bool = False,
        masking_token_emb: bool = False,
        masking_token_index: str = None,
        calcaluate_distance: bool = False,
        TEB_max_steps: int = 20,
        **kwds):

        callback = kwds.pop("callback", None)   # callback = None
        callback_steps = kwds.pop("callback_steps", None)   #  callback_steps = None

        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE), output_hidden_states=False, output_attentions=False)[0]

        if concat_pure_text_emb:
            vowels = 'aeiou'
            given_prompt_len = len(self.tokenizer.encode(prompt[0]))
            pure_embedding_list = []
            words = prompt[0].split()
            for indice in indices_to_balance:
                indice -= 1
                chosen_word = words[indice]
                if chosen_word.lower() in vowels:
                    article = "an"
                else:
                    article = "a"
                
                separate_prompt = "A photo of {} {}".format(article, chosen_word)
                # print("separate_prompt = ", separate_prompt)
                separate_text_input = self.tokenizer(separate_prompt, padding="max_length", max_length=77, return_tensors="pt")
                pure_embedding_list.append(self.text_encoder(separate_text_input.input_ids.to(DEVICE), output_hidden_states=False, output_attentions=False)[0])

            # text_embeddings[:, 1:3, :] = pure_embedding_list[0][:, 4:6, :]
            text_embeddings[:, 4:6, :] = pure_embedding_list[1][:, 4:6, :]
        
        if text_emb_optimize:
            loss_log = open(os.path.join(save_path, '_text_emb_optimize_log.txt'), 'a')
            loss_log.write("========== {}_{} =========\n".format(prompt[0], seed))
            # print("prompt[0] = ", prompt[0])
            vowels = 'aeiou'
            given_prompt_len = len(self.tokenizer.encode(prompt[0]))
            pure_embedding_list = []
            words = prompt[0].split()
            for indice in indices_to_balance:
                indice -= 1
                chosen_word = words[indice]
                if chosen_word.lower() in vowels:
                    article = "an"
                else:
                    article = "a"
                
                separate_prompt = "A photo of {} {}".format(article, chosen_word)
                separate_text_input = self.tokenizer(separate_prompt, padding="max_length", max_length=77, return_tensors="pt")
                pure_embedding_list.append(self.text_encoder(separate_text_input.input_ids.to(DEVICE), output_hidden_states=False, output_attentions=False)[0][:, 5, :])
            text_embeddings = self._textEMB_step(text_embeddings, pure_embedding_list, indices_to_balance=indices_to_balance, given_prompt_len=given_prompt_len, step_size=20, max_opt_iter=TEB_max_steps, loss_log=loss_log)

        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE), output_hidden_states=False)[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)

        # control for analysis
        if masking_token_emb:
            encoder_attention_mask = torch.ones(text_embeddings.size(0), text_embeddings.size(1)).to(DEVICE)
            start_idx, end_idx = masking_token_index
            encoder_attention_mask[1, start_idx:end_idx] = 0
        else:
            encoder_attention_mask = None
        
        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 

            noise_pred = self.unet(model_inputs, t, 
                                   encoder_hidden_states=text_embeddings,
                                   encoder_attention_mask=encoder_attention_mask
                                   ).sample
            
            if calcaluate_distance:
                # calculate the cross-map similarity and text embedding similarity
                if i == 0:
                    if text_emb_optimize: 
                        status = "opt"
                    else: 
                        status = "default"
                    self._cal_textEMB_sim(text_embeddings, save_path, prompt[0], seed, status=status,
                                            indices_to_balance=indices_to_balance,
                                            attention_store=attention_store)
                    break

            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                text_embeddings = callback_outputs.pop("prompt_embeds", text_embeddings)
                unconditional_embeddings = callback_outputs.pop("negative_prompt_embeds", unconditional_embeddings)

            # call the callback, if provided
            if i == len(self.scheduler.timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        image = self.latent2image(latents, return_type="pt")
        image, has_nsfw_concept = self.run_safety_checker(image, DEVICE, text_embeddings.dtype)

        return image, has_nsfw_concept