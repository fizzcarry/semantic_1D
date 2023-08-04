
if __name__ == "__main__":
    net = SETRModel(patch_size=(32, 32),
                    in_channels=3,
                    out_channels=1,
                    hidden_size=1024,
                    sample_rate=5,
                    num_hidden_layers=1,
                    num_attention_heads=16,
                    decode_features=[512, 256, 128, 64])
    t1 = torch.rand(1, 3, 512, 512)
    print("input: " + str(t1.shape))