def plot_perf(hist):
    fig, ax = plt.subplots(ncols = 3, figsize = (20, 5))

    ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
    ax[0].plot(hist.history['val_total_loss'], color='c', label = 'validation loss')
    ax[0].title.set_text('Loss')
    ax[0].legend()

    ax[1].plot(hist.history['class_loss'], color= 'blue', label='class loss')
    ax[1].plot(hist.history['val_class_loss'], color = 'orange', label =' Validation Class loss')
    ax[1].title.set_text('Classification Loss')
    ax[1].legend()

    ax[2].plot(hist.history['regress_loss'], color = 'teal', label = 'regress loss')
    ax[2].plot(hist.history['val_regress_loss'], color = 'red', label = 'validation Regression Loss')
    ax[2].title.set_text("Regression Loss")
    ax[2].legend()

    plt.show()