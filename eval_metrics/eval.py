import os
import argparse
from owlv2 import owlv2_eval

if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(description="Demo for the attention visualization")
    parser.add_argument("--save_path", type=str, default="./mixture_eval", help="Path to the output directory")
    parser.add_argument("--src_dir", type=str, default="../result/test_sample", help="Path to the src images")
    
    args = parser.parse_args()
    exp_name= os.path.basename(args.src_dir)
    
    print("{} | eval path: {}".format(exp_name, args.src_dir))

    summary = owlv2_eval(exp_name=exp_name, image_path=args.src_dir, save_path=args.save_path)
    print(" ===== summary of prediction ===== ")
    for key, value in summary['identity_counts'].items():
        print("{:23s} {}".format(key, value))
