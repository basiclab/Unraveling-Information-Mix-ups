import json
import random

def load_animals(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['animals']

def gen_prompt_txt(candidate_data, save_json, num_prompts, num_objects):

    # Load animals and generate prompts
    animals_list = load_animals(candidate_data)

    # Setting the seed for the random number generator for reproducibility
    random.seed()

    output_file = open(save_json, 'w')
    prompts_count = 0
    while prompts_count < num_prompts:
        sample_list = random.sample(animals_list, num_objects)
        prompt = ""
        for idx, animal in enumerate(sample_list):
            prefix = "an" if animal == "elephant" else "a"
            if idx == 0:
                prompt += "{} {}".format(prefix, animal)
            else:
                prompt += " and {} {}".format(prefix, animal)
        # Generating a random seed within a typical range for 32-bit integers
        random_seed = random.randint(0, 2**32 - 1)
        output_file.write(f"{random_seed};{prompt}\n")
        prompts_count += 1


if __name__ == "__main__":
    candidate_data = "prompt_candidate.json"
    num_prompts = 1000
    num_objs = 2
    save_json = "prompts{}_{}objs_2.txt".format(num_prompts, num_objs)
    gen_prompt_txt(candidate_data, save_json, num_prompts, num_objs)