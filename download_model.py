import os

def main():
    if not os.path.exists("./checkpoints/test_local/"):
        os.makedirs("./checkpoints/test_local/", exist_ok=True)
        print("Downloading model...")
        os.system("wget http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth -P ./checkpoints/test_local/")

if __name__ == "__main__":
    main()