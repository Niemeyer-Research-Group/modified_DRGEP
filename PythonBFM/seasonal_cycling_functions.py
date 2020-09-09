import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import brewer2mpl

def calc_day_of_year(time_seconds):
    """ function that calculates the day of year """
    
    cycle = 366.0
    
    # calculate day of year
    if time_seconds == 0.0:
        day_of_year = 1.0
    else:
        sec_per_day = 86400.0
        time_day = time_seconds/sec_per_day
        day_of_year = numpy.floor(time_day) + 1
        
    # calculate year
    year = numpy.ceil(day_of_year/cycle)
    
    # correct day of year if year>1
    if year > 1:
        day_of_year -= (year - 1)*cycle

    return day_of_year

def calc_fraction_of_day(time_seconds):
    """ function that calculates the fration of the day """
    
    sec_per_day = 86400.0
    seconds_of_day = numpy.fmod(time_seconds,sec_per_day)
    fraction_of_day = seconds_of_day/sec_per_day

    return fraction_of_day


def get_wind(time,w_win,w_sum):
    """ function that calculates the seasonal wind values """

    day_of_year = calc_day_of_year(time)
    fraction_of_day = calc_fraction_of_day(time)
    wind = (w_sum+w_win)/2 - ((w_sum-w_win)/2)*numpy.cos((day_of_year+(fraction_of_day - 0.5))*(numpy.pi/180))

    return wind


def get_salinity(time,s_win,s_sum):
    """ function that calculates the seasonal salinity values """

    day_of_year = calc_day_of_year(time)
    fraction_of_day = calc_fraction_of_day(time)
    salinity = (s_sum+s_win)/2.0 - ((s_sum-s_win)/2.0)*numpy.cos((day_of_year+(fraction_of_day - 0.5))*(numpy.pi/180))

    return salinity


def get_sunlight(time,q_win,q_sum):
    """ function that calculates the sunlight """

    day_of_year = calc_day_of_year(time)
    fraction_of_day = calc_fraction_of_day(time)
    latitude = 45.0
    light = (q_sum+q_win)/2.0 - (q_sum-q_win)/2.0*numpy.cos(day_of_year*(numpy.pi/180))
    cycle = 360
    declination = -0.406*numpy.cos(2.0*numpy.pi*int(day_of_year)/cycle)
    day_length = numpy.arccos(-numpy.tan(declination)*numpy.tan(latitude*(numpy.pi/180)))/numpy.pi*24.0
#    print(day_length)
    day_time = fraction_of_day*24.0
    day_time = numpy.abs(day_time - 12.0)
    day_len = day_length/2.0
    if(day_time<day_len):
        day_time = day_time/day_len*numpy.pi
        wlight = light*numpy.cos(day_time) + light
    else:
        wlight = 0.0
    
    return wlight

def get_temperature(time,t_win,t_sum,tde):
    """ function that calculates the seasonal temperature """

    day_of_year = calc_day_of_year(time)
    fraction_of_day = calc_fraction_of_day(time)
    temperature = (t_sum + t_win)/2.0 - (t_sum - t_win)/2.0*numpy.cos((day_of_year+(fraction_of_day - 0.5))*(numpy.pi/180)) - tde*0.5*numpy.cos(2*numpy.pi*fraction_of_day)

    return temperature

def calculate_density(temperature, salinity, depth):
    """ This function computes the density in kg m^-3
    
        obtained from envforcing.F90 density function which is orginally from 
        Mellor, 1991, J. Atmos. Oceanic Tech., 609-611
    """

    grav = 9.806
    
    # density is computed at the middle of the layer
    depth = depth/2
    
    # approximate pressure in units of bars
    p = -grav*1.025*depth*0.01
    cr = 1449.1 + 0.0821*p + 4.55*temperature - 0.045*(temperature**2) + 1.34*(salinity - 35.0)
    cr = p/(cr**2)

    # calculate density
    density = (999.842594 + 6.793952E-2*temperature - 9.095290E-3*(temperature**2) + 
               1.001685E-4*(temperature**3) - 1.120083E-6*(temperature**4) + 
               6.536332E-9*(temperature**5) + 
               (0.824493 - 4.0899E-3*temperature + 7.6438E-5*(temperature**2) - 
                8.2467E-7*(temperature**3) + 5.3875E-9*(temperature**4))*salinity +
                (-5.72466E-3 + 1.0227E-4*(temperature) - 1.6546E-6*(temperature**2))*(salinity**1.5) +
                4.8314E-4*(salinity**2)) + 1.0E+5*cr*(1.0 - (cr + cr))

    return density

if __name__ == '__main__':
    # Wind
    w_win = 20.0            # Winter wind speed
    w_sum = 10.0            # Summer wind speed
    wind = []               # set up list for wind values
    
    # Salinity
    s_win = 37.0            # Winter salinity value
    s_sum = 34.0            # Summer salinity value
    salinity = []           # set up list for salinity values
    
    # Short wave irradiance flux (W/m^2)
    q_win = 20.0           # Winter irradiance value
    q_sum = 300.0          # Summer irradiance value
    light = []              # set up list for light values
    
    # Temperature
    t_win = 8.0             # Winter temp value
    t_sum = 28.0            # Summer temp value
    tde = 1.0               # Sinusoidal temperature daily excursion degC
    temperature = []
    
    # day of year
    day_of_year = numpy.zeros(7310)
#    time_values = numpy.linspace(0,86400*360*2,721)
#    time_values = numpy.linspace(8640, 43200, 5)
#    time_values = numpy.linspace(8640, 86400*365, 5000)
    time_values = numpy.linspace(8640, 6.31584e+07, 7310)
#    time_values = numpy.insert(t_span, 0, 0)
    for time in time_values:
        wind.append(get_wind(time,w_win,w_sum))
        salinity.append(get_salinity(time,s_win,s_sum))
        light.append(get_sunlight(time,q_win,q_sum))
        temperature.append(get_temperature(time,t_win,t_sum,tde))
    
    for i,time in enumerate(time_values):
        day_of_year[i] = calc_day_of_year(time)

    # Make pdf containg plots for the seasonal cycling
    bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
    colors = bmap.mpl_colors
    with PdfPages('seasonal_cycling_plots.pdf') as pdf:
#        # plot wind
#        fig = plt.figure(1)
#        plt.axes(frameon=0)
#        plt.grid(axis='y', color="0.9",linestyle='--')
#        plt.plot(time_values, wind, color=colors[1])
#        plt.ylabel('Wind', fontsize=14)
#        plt.xlabel('Time (sec)', fontsize=14)
#        pdf.savefig(fig, bbox_inches = "tight")
#        
#        # Plot salinity 
#        fig = plt.figure(2)
#        plt.axes(frameon=0)
#        plt.grid(axis='y', color="0.9",linestyle='--')
#        plt.plot(time_values, salinity, color=colors[2])
#        plt.yticks(numpy.arange(min(salinity), max(salinity)+1, 1))
#        plt.ylabel('Salinity', fontsize=14)
#        plt.xlabel('Time (sec)', fontsize=14)
#        pdf.savefig(fig, bbox_inches = "tight")
#        
#        # Plot light
        fig = plt.figure(3)
        plt.axes(frameon=0)
        plt.grid(axis='y', color="0.9",linestyle='--')
        plt.plot(time_values, light, "o-", color=colors[3])
        plt.ylabel('Light', fontsize=14)
        plt.xlabel('Time (sec)', fontsize=14)
        pdf.savefig(fig, bbox_inches = "tight")
#        
#        # Plot temperature
#        fig = plt.figure(4)
#        plt.axes(frameon=0)
#        plt.grid(axis='y', color="0.9",linestyle='--')
#        plt.plot(time_values, temperature, color=colors[4])
#        plt.yticks(numpy.arange(min(temperature), max(temperature)+1, 4))
#        plt.ylabel('Temperature', fontsize=14)
#        plt.xlabel('Time (sec)', fontsize=14)
#        pdf.savefig(fig, bbox_inches = "tight")
