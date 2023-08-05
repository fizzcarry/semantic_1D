from SETR.transformer_seg import SETRModel, Vit
import torch
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
import matplotlib.pyplot as plt
if __name__ == "__main__":
    net = SETRModel(patch_size=(32, 32), 
                    in_channels=3, 
                    out_channels=1, 
                    hidden_size=1024, 
                    sample_rate=5,
                    num_hidden_layers=1, 
                    num_attention_heads=16, 
                    decode_features=[512, 256, 128, 64])
    img = read_image(r"D:\all_code\cam_code\data\image/1.png")
    img = img[:3, :, :]
    input_tensor = normalize(resize(img, [224, 224]) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t1=input_tensor.unsqueeze(0)

    print("input: " + str(t1.shape))
    out=net(t1).detach().numpy()
    print("output: " + str(out.shape))
    from torchcam.utils import overlay_mask
    result = overlay_mask(to_pil_image(img), to_pil_image(out[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result);
    plt.axis('off');
    plt.tight_layout();
    plt.show()

    model = Vit(patch_size=(32, 32), 
                    in_channels=1, 
                    out_class=10, 
                    sample_rate=4,
                    hidden_size=1024, 
                    num_hidden_layers=1, 
                    num_attention_heads=16)
    
    t1 = torch.rand(1, 1, 512, 512)
    print("input: " + str(t1.shape))

    print("output: " + str(model(t1).shape))

