import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
# from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--input_dir', required=True,
                        help='Path to the folder with images inputs')
    parser.add_argument('--output_dir', required=True,
                        help='Path to the folder to save the image outputs')
    args = parser.parse_args()
    return args

def inference(model, input_height, input_width, input_dir, output_dir):
    total_loss =0 
    toal_count = 0
    print("============================= INFERENCE ============================")
    model.switch_to_eval()

    for img_name in os.listdir(input_dir):
        img_path = input_dir + img_name

        img = np.float32(io.imread(img_path))/255.0
        img = resize(img, (input_height, input_width), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda() )
        pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
        pred_inv_depth = 1/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        # you might also use percentile for better visualization
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

        img_output_path = output_dir + img_name.split('.')[0]
        io.imsave(f"{img_output_path}.png", pred_inv_depth)
        # print(pred_inv_depth.shape)
    sys.exit()


def main():
    opt = TrainOptions().parse()
    model = create_model(opt)
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    input_height = 384
    input_width  = 512

    inference(model, input_height, input_width, args.input_dir, args.output_dir)
    print("We are done")

if __name__ == "__main__":
    main()