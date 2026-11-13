COLORS = {"Gryffindor": "red", "Hufflepuff": "gold", "Ravenclaw": "blue", "Slytherin": "green"}


def by_house(values, houses, house):
    return [v for v, h in zip(values, houses) if h == house and v is not None]


def show_or_save(plt, argv):
    """Show the figure, or save it to argv[2] when given (useful headless)."""
    if len(argv) > 2:
        plt.savefig(argv[2], dpi=200)
    else:
        plt.show()
