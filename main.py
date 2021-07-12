from trainmodel import *

colors = ["RGB", "YUV"]

for color in colors:
    trained_model = load_model(color)
    r_psnr = 0
    p_psnr = 0
    r_ssim = 0
    p_ssim = 0
    for idx in range(10):
        resized_psnr, pred_psnr, resized_ssim, pred_ssim = resized_output_psnr_ssim(
            idx, color, trained_model)
        r_psnr += resized_psnr
        p_psnr += pred_psnr
        r_ssim += resized_ssim
        p_ssim += pred_ssim
    r_psnr /= 10.0
    p_psnr /= 10.0
    r_ssim /= 10.0
    p_ssim /= 10.0

    print(color)
    tf.print("Average of psnr between resized image and groundtruth: ", r_psnr)
    tf.print("Average of psnr between predicted image and groundtruth: ", p_psnr)
    tf.print("Average of ssim between resized image and groundtruth: ", r_ssim)
    tf.print("Average of ssim between predicted image and groundtruth: ", p_ssim)

