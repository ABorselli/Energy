import pandas as pd
import numpy as np

# First, we load the energy data from the file `Energy Indicators.xls`,
# which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls)
# from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls)
# for the year 2013, and should be put into a DataFrame with the variable name of **Energy**.
#
# We relabel the columns as the following:
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable]`
#
# Then, we convert `Energy Supply` to gigajoules (**Note: there are 1,000,000 gigajoules in a petajoule**).
# For all countries which have missing data (e.g. data with "..."), this is reflected as `np.NaN` values.
#
# For merging datasets, we rename the following list of countries:
#
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
#
# There are also several countries with numbers and/or parenthesis in their name.
# We remove these, e.g. `'Bolivia (Plurinational State of)'` should be `'Bolivia'`.
# `'Switzerland17'` should be `'Switzerland'`.
#
# Next, we load the GDP data from the file `world_bank.csv`,
# which is a csv containing countries' GDP from 1960 to 2015
# from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). We call this DataFrame **GDP**.
#
# We rename the following list of countries:
#
# ```"Korea, Rep.": "South Korea",
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
#
# Finally, we load the [Sciamgo Journal and Country Rank data for Energy Engineering
# and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`,
# which ranks countries based on their journal contributions in the aforementioned area.
# We call this DataFrame **ScimEn**.
#
# Then, we join the three datasets: GDP, Energy, and ScimEn into a new dataset
# (using the intersection of country names). We will use only the last 10 years (2006-2015) of GDP data
# and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15).
#
# The index of this DataFrame should be the name of the country,
# and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].

Energy = pd.read_excel('~/Downloads/Energy Indicators.xls', header=16, skipfooter=38)
Energy=Energy.drop(labels=0, axis=0)
Energy=Energy.drop(columns=Energy.columns[0])
Energy=Energy.drop(columns=Energy.columns[0])
Energy.rename(columns={'Unnamed: 2': 'Country', 'Energy Supply per capita': 'Energy Supply per Capita', 'Renewable Electricity Production': '% Renewable'}, inplace=True)
Energy['Country'].replace('\d+', '', regex=True, inplace=True)
Energy['Country'].replace('[ ]\(.*', '', regex=True, inplace=True)
Energy.replace({"Republic of Korea": "South Korea",
                "United States of America": "United States",
                "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                "China, Hong Kong Special Administrative Region": "Hong Kong"},
               inplace=True)
Energy['Energy Supply']= Energy['Energy Supply']*1000000
GDP = pd.read_csv('~/Downloads/world_bank.csv', header=4)
GDP.rename(columns={'Country Name': 'Country'}, inplace=True)
GDP.replace({"Korea, Rep.": "South Korea",
             "Iran, Islamic Rep.": "Iran",
             "Hong Kong SAR, China": "Hong Kong"},
            inplace=True)
ScimEn = pd.read_excel('~/Downloads/scimagojr-3.xlsx')
DataFrame = pd.merge(pd.merge(ScimEn, Energy, how='inner', on='Country'), GDP, how='inner', on='Country')
DataFrame = DataFrame.set_index('Country')
columns_to_keep = ['Rank', 'Documents', 'Citable documents', 'Citations',
                   'Self-citations', 'Citations per document', 'H index',
                   'Energy Supply', 'Energy Supply per Capita', '% Renewable',
                   '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                   '2014', '2015']
DataFrame = DataFrame[columns_to_keep]
DataFrame = DataFrame.iloc[0:15]
print(DataFrame)

# To see how many observations we lost, we can compare the top 15 to the prior dataset:

DataFrame2 = pd.merge(pd.merge(ScimEn, Energy, how='outer', on='Country'), GDP, how='outer', on='Country')
print(len(DataFrame2) - len(DataFrame))

# We can see the top 15 countries for average GDP over the last 10 years using the following function:

cols = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
DataFrame['avgGDP'] = DataFrame[cols].mean(axis=1)
avgGDP = DataFrame.loc[:,'avgGDP'].sort_values(ascending=False)
print(avgGDP)

# Now, we can can check how much the GDP had changed over the 10 year span
# for the country with the 6th largest average GDP.

DataFrame['GDP change'] = DataFrame['2015']-DataFrame['2006']
print(DataFrame['GDP change'].loc['United Kingdom'])

# What is the mean energy supply per capita?

DataFrame['Energy Supply per Capita'].mean()

# What country has the maximum % Renewable and what is the percentage?

max_percent = DataFrame['% Renewable'].max()
print((DataFrame.index[DataFrame['% Renewable']==max_percent][0], max_percent))

# Next, we create a new column that is the ratio of Self-Citations to Total Citations.
# What is the maximum value for this new column, and what country has the highest ratio?

DataFrame['Citation Ratio'] = DataFrame['Self-citations']/DataFrame['Citations']
max_ratio = DataFrame['Citation Ratio'].max()
print((DataFrame.index[DataFrame['Citation Ratio']==max_ratio][0], max_ratio))

# Now, we create a column that estimates the population using Energy Supply and Energy Supply per capita.
# What is the third most populous country according to this estimate?

DataFrame['Pop_est'] = DataFrame['Energy Supply']/DataFrame['Energy Supply per Capita']
sorted_pops = DataFrame['Pop_est'].sort_values(ascending=False)
print(sorted_pops[sorted_pops==sorted_pops[2]].index[0])

# Next, we create a column that estimates the number of citable documents per person.
# What is the correlation between the number of citable documents per capita and the energy supply per capita?
# We use the `.corr()` method, (Pearson's correlation).

DataFrame['CDpP'] = np.float64(DataFrame['Citable documents']/DataFrame['Pop_est'])
DataFrame['Energy Supply per Capita'] = np.float64(DataFrame['Energy Supply per Capita'])
print(DataFrame['Energy Supply per Capita'].corr(DataFrame['CDpP']))

# Then, we create a new column with a 1 if the country's % Renewable value is at or above the median
# for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.

#med_percent = np.nanmedian(DataFrame['% Renewable'])
#DataFrame['HighRenew'] = DataFrame['% Renewable']
#for index in range(0,15):
#    if DataFrame['% Renewable'].iloc[index]>=med_percent:
#        DataFrame['HighRenew'].iloc[index] = 1
#    else:
#        DataFrame['HighRenew'].iloc[index] = 0
#HighRenew = DataFrame['HighRenew']
#print(HighRenew)

# Finally, we create a DataFrame that displays the sample size (the number of countries in each continent bin),
# and the sum, mean, and std deviation for the estimated population of each country.

ContinentDict  = {'China':'Asia', 'United States':'North America', 'Japan':'Asia',
                  'United Kingdom':'Europe', 'Russian Federation':'Europe',
                  'Canada':'North America', 'Germany':'Europe', 'India':'Asia',
                  'France':'Europe', 'South Korea':'Asia', 'Italy':'Europe',
                  'Spain':'Europe', 'Iran':'Asia', 'Australia':'Australia',
                  'Brazil':'South America'}
DataFrame.reset_index(inplace=True)
DataFrame['Continent'] = DataFrame['Country'].map(ContinentDict)
DataFrame['Pop_est'] = np.float64(DataFrame['Pop_est'])
df = DataFrame.pivot_table(values='Pop_est', index='Continent',
                           aggfunc=[np.count_nonzero, np.sum, np.mean, np.std],
                           margins=True)
df = df.drop(labels='All', axis=0)
df.columns=['size', 'sum', 'mean', 'std']
print(df)

#def plot_example():
#    import matplotlib as plt
#    get_ipython().run_line_magic('matplotlib', 'inline')
#    Top15 = Data_Frame_Build()
#    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter',
#                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
#                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'],
#                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);
#
#    for i, txt in enumerate(Top15.index):
#        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')
#
#    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")
#
#plot_example()
