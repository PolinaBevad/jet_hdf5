import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT_PROPERTIES_TTF_FILE = 'FreeSansBold.ttf'

# Unicodes for beautiful symbols of  Lorentz factor and b=v/c
G = u'\u0393'
B = u'\u03B2'
pho = u'\u03C1'
LIGHT_SPEED = 29979245800.0


def main():
    # Font properties to show unicode symbols on plot
    global prop
    prop = FontProperties()
    prop.set_file(FONT_PROPERTIES_TTF_FILE)

    rad, dens, pres = read_model()
    plot_density(rad, dens)
    plot_pressure(rad, pres)


def read_model():
    file = "data\\16TI_rad_dens_pres.txt"
    rad = []
    dens = []
    pres = []
    with open(file, encoding='utf-16-le') as f:
        for line in f:
            string = line.split()
            if "index" in string[0]:
                continue
            rad.append(float(string[1]))
            dens.append(float(string[2]))
            pres.append(float(string[3])/LIGHT_SPEED**2)
    return rad, dens, pres


def plot_density(rad, dens):
    # default style for plot
    plt.rcdefaults()
    plt.style.use('ggplot')
    plt.plot(rad, dens)

    # Labels and logarithmic scale
    plt.xlabel(f'log(R)[см]', fontproperties=prop)
    plt.ylabel(f'log({pho})[г/см3]', fontproperties=prop)
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Распределение плотности вдоль радиуса в модели 16TI')
    plt.show()


def plot_pressure(rad, pres):
    # default style for plot
    plt.rcdefaults()
    plt.style.use('ggplot')
    plt.plot(rad, pres)

    # Labels and logarithmic scale
    plt.xlabel(f'log(R)[см]', fontproperties=prop)
    plt.ylabel(f'log(P)[эрг/см3]', fontproperties=prop)

    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Распределение давления вдоль радиуса в модели 16TI')
    plt.show()


if __name__ == '__main__':
    main()
