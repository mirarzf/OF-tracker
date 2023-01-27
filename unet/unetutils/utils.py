import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 and mask.shape[0] > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_img_and_mask_and_gt(img, gt, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 and mask.shape[0] > 2 else 1
    fig, ax = plt.subplots(1, classes + 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Ground Truth')
    ax[1].imshow(gt)
    if classes > 1:
        for i in range(classes):
            ax[i + 2].set_title(f'Output mask (class {i + 1})')
            ax[i + 2].imshow(mask[i, :, :])
    else:
        ax[2].set_title(f'Output mask')
        ax[2].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
