import matplotlib.pyplot as plt


def plot_sample(sample):
    image = sample[0]
    label = sample[1]
    boxes = sample[2]

    plt.imshow(image)

    # plot the boxes

    img_width, img_height = image.size
    for box in boxes:
        x_min, y_min, x_max, y_max = (
            box["x_min"] * img_width,
            box["y_min"] * img_height,
            box["x_max"] * img_width,
            box["y_max"] * img_height,
        )
        plt.plot(
            [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min]
        )
        plt.text(x_min, y_min, label)
    plt.show()


def plot_prediction(predictions, sample):
    image = sample[0]
    label = sample[1]
    boxes = sample[2]
    predicted_boxes = predictions["objects"]
    fig, ax = plt.subplots()
    ax.imshow(image)
    img_width, img_height = image.size

    # for box in boxes:
    #   x_min, y_min, x_max, y_max = box['x_min'] * img_width, box['y_min'] * img_height, box['x_max'] * img_width, box['y_max'] * img_height
    #  plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min])
    # plt.text(x_min, y_min, label)

    for box in predicted_boxes:
        x_min, y_min, x_max, y_max = (
            box["x_min"] * img_width,
            box["y_min"] * img_height,
            box["x_max"] * img_width,
            box["y_max"] * img_height,
        )
        ax.plot(
            [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min]
        )
        ax.text(x_min, y_min, label)

    # return the plot as a figure
    return fig
