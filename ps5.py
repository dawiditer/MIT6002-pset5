# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    models = []
    #for each degree
    for degree in degs:
        #get the coefficients of the polynomial
        coefficients = pylab.polyfit(x, y, degree)
        models.append(coefficients)
    return models


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    #numerator
    estimatedError = ((y - estimated)**2).sum()
    #denom
    y_mean = y.mean()
    variability = ((y - y_mean)**2).sum()
    
    return 1.0 - estimatedError/variability

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        _plot_model(x, y, model, r_squared)

## Helper method ##
def _plot_model(x, y, model, eval_function):
    """
    Plots a model as specified in evaluate_models_on_training
    """
    #we need the y vals generated by the model, ie estimated
    estimated = pylab.polyval(model, x)
    #we then use this to generate r_squared
    eval_ = eval_function(y, estimated)
    
    #degree of the model
    deg = len(model) - 1    
    deg_str = (["1st", "2nd", "3rd"])[deg - 1] if deg < 4 else str(deg) + "th"
    title = "Temp changes per year (" + deg_str + " degree polynom)"
    
    func_str = "R**2" if eval_function is r_squared else "RMSE"
    
    title += "\n" + func_str + " = {:.4}".format(eval_)
    
    if func_str == "R**2" and deg == 1:
        standard_error = se_over_slope(x, y, estimated, model)
        title += ", SE over slope = {:.3}".format(standard_error)
    
    pylab.title(title)
    pylab.plot(x, y, 'bo')
    pylab.plot(x, estimated, 'r')
    pylab.ylabel("Temperatures(celcius)")
    pylab.xlabel("Years")
    pylab.show()
## end of helper method ##
        
    
def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    #get the data for each city for each year
    samples = []
    num_cities = len(multi_cities)
    for year in years:
        national_total_temp = 0
        for city in multi_cities:
            avg_temp = climate.get_yearly_temp(city, year).mean()
            national_total_temp += avg_temp
        samples.append(national_total_temp/num_cities)
    
    return pylab.array(samples)
    

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """        
    #function to map all y elements to moving average
    avg_map = lambda i: pylab.mean(y[max(0, i-window_length+1):i+1])
    
    return pylab.array([avg_map(i) for i in range(len(y))])

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    #numrator
    estimation_error = ((y - estimated)**2).sum()
    #denom
    sample_size = len(y)
    
    return pylab.sqrt(estimation_error/sample_size)

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    #Calculate a temperature for each day in that year, 
    #by averaging the temperatures for that day across the specified cities.
    annual_stds = []
    #Get an array holding the average temperatures for each day
    for year in years:
        #total temperatures for each city
        sums_of_temp = pylab.array([])
        for city in multi_cities:
            city_annual_temps = climate.get_yearly_temp(city, year)
            if not sums_of_temp.any():
                sums_of_temp = pylab.array([0]*len(city_annual_temps))
                
            sums_of_temp = sums_of_temp + city_annual_temps
            
        #daily average temperatures 
        avgs_annual_temps = sums_of_temp/len(multi_cities) 
        #standard deviation
        std = pylab.std(avgs_annual_temps)
        annual_stds.append(std)
    
    return pylab.array(annual_stds)        
        
def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        _plot_model(x, y, model, rmse)

if __name__ == '__main__':
    # Part A.4
    climate = Climate("data.csv")
    
    def problem4_i():
        samples = []
        month = 1
        day = 10    
        city = 'NEW YORK'
        for year in TRAINING_INTERVAL:
            samples.append(climate.get_daily_temp(city, month, day, year))
        return samples
                
    def problem4_ii():
        #data samples are the averages for each year
        samples = []
        city = 'NEW YORK'
        for year in TRAINING_INTERVAL:
            avg_temp = climate.get_yearly_temp(city, year).mean()
            samples.append(avg_temp)
        return samples
    
    ## Helper method ##
    def plot_training(samples, years, deg):
        xvals = pylab.array(years)
        yvals = pylab.array(samples)
        models = generate_models(xvals, yvals, deg)
        evaluate_models_on_training(xvals, yvals, models)
        
        return models
    ## end og helper method ##
    
#    plot_training(problem4_i(), TRAINING_INTERVAL, [1])
#    plot_training(problem4_ii(), TRAINING_INTERVAL, [1])
        
    # Part B
#    samples = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
#    plot_training(samples, TRAINING_INTERVAL, [1])

    # Part C
#    national_temps = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
#    
#    samples = moving_average(national_temps, 5)
#    plot_training(samples, TRAINING_INTERVAL, [1])

    # Part D.2
#    national_training_temps = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
#    training_samples = moving_average(national_training_temps, 5)
#    models = plot_training(training_samples, TRAINING_INTERVAL, [1, 2, 20])
#    national_testing_temps = gen_cities_avg(climate, CITIES, TESTING_INTERVAL)
#    xvals = pylab.array(TESTING_INTERVAL)
#    yvals = pylab.array(national_testing_temps)
#    
#    evaluate_models_on_testing(xvals, yvals, models)
        
    # Part E
    stds = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    training_samples = moving_average(stds, 5)
    plot_training(training_samples, TRAINING_INTERVAL, [1])