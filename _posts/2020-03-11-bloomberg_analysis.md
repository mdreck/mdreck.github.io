---
title: "Bloomberg Automotive Analysis"
date: 2020-03-11
tags: [data analysis, data science, linear regression]
header:
  image: "/images/automobile-production.png" 
excerpt: "Automotive screening model to determine over/undervaluation of auto manufacturers and suppliers"
mathjax: "true"
---
- [Bloomberg Automotive Analysis](https://github.com/mdreck/mdreck.github.io/blob/master/bloomberg_analysis/bloomberg_analysis.xlsx): Click link and download Excel file, or view PDF in link below.

- [Bloomberg Automotive Analysis (PDF)](https://github.com/mdreck/mdreck.github.io/blob/master/bloomberg_analysis/bloomberg_analysis.pdf): Click link to view PDF.

# Bloomberg Automotive Screening Model
 This Bloomberg automotive screening model performs a regression on automotive industry data to determine whether auto suppliers and manufacturers are being undervalued or overvalued. The data was downloaded using a Bloomberg Terminal.
 
In Bloomberg I used the ‘RAY 3000 Index’ command to pull up the Russell 3000 Index and deleted all columns except the Ticker and Name columns. I added columns by typing GICS in the fields search bar and added the following: Sector Name, Industry Name, Industry Group Name, Sub-Industry Name. The Russell 3000 index data was then exported from Bloomberg into excel.
In excel, I sorted the Russell 3000 Index by GICS Sub-Industry Name and performed the following: 
- Removed all companies (rows) that did not include the Sub-Industry ‘Auto Parts and Equipment’ or ‘Automobile Manufacturers’
- Removed all columns other than the ‘Ticker’ and ‘Name’ columns and sort data A to Z

I used the Bloomberg excel add-in to add fundamental financial data related to growth, profitability and risk, and compared those factors to a valuation metric (The Bloomberg Excel add-in is only available when using a Bloomberg machine). 
Use ‘Find Fields’ under the Bloomberg add-in tab to add financial data. I selected the following fields for my analysis:
- Independent Variables:
  - ‘SALES_3YR_AVG_GROWTH’ and ‘GROSS_MARGIN’ to measure of growth and profitability 
  - ‘TOT_DEBT_TO_EBITDA’ (Debt / EBITDA) is included as a factor of risk and measures a company’s debt exposure. EBITDA is a company’s earnings before interest, tax, depreciation, and amortization
- Dependent Variables:
  - ‘EV_TO_T12M_EBITDA’ (Enterprise Value / EBITDA) is a valuation metric that represents the fair market value of a company. Enterprise value = mkt cap + preferred shares + minority interest + debt – total cash

Once the column fields were populated, I used the BDP function to pull data for each company. The functions are viewable on the hidden ‘Bloomberg Data’ sheet within the excel workbook (the values will not populate unless you are on a Bloomberg Terminal, so I copied and pasted that data as values in the ‘Model’ sheet).

I’ve selected these measurements because it is reasonable to expect that companies with high sales growth, high gross margins, and low debt would have a higher valuation than companies with poor sales growth and gross margins, and high debt. I performed a regression to evaluate the performance of the stock screening model and determine if the independent variables are good predictors of the market valuation (dependent variable). Perform the regression using the Data Analysis ToolPak. Use the market valuation (EV_TO12M_EBITDA) as the ‘Input Y Range’ and the independent variables from above as the ‘Input X Range’.
 
The regression output is included on the ‘Model’ sheet
These are the ‘goodness of fit’ measures. The most important of them for our purposes is the adjusted R square statistic. It determines the predictive value of the independent variables on the dependent variable. This indicates that 36% of the variation in the dependent variable can be explained by the independent variables. In other words, 36% of the values fit the model, which is not a particularly strong relationship but at least it’s positive. The Multiple R statistic is the correlation coefficient and describes how strong the linear relationship is, so there is a decently positive linear relationship. 
 
Looking at the t-stat and p-value statistics, we can determine that the DEBT to EBITDA variable is not a particularly strong predictor compared to Sales Growth and Gross Margin. Ideally, we want t-stat values greater than 2.00 and p-values of less than 5% (0.05). The coefficients column provides the slope of the variable. Sales growth and gross margin are both positive slopes as we’d expect companies with good growth and high profits to have a higher valuation than those with poor growth and low profits. We would assume that companies with high debt exposure would be valued lower than those with less debt, however, our regression found that companies with high debt exposure are valued with a positive slope. 

Now that we have the intercept and slopes, we can forecast what the company should be valued at according to the linear regression equation (y = mx + b). The values and function are displayed in the ‘Model’ sheet under the ‘Forecast EV to EBITDA’ column. From the forecast, we can identify companies that are overvalued or undervalued compared to their current valuation. Companies with forecasts that are higher than their current valuation could be considered undervalued, whereas companies with a lower forecast could be considered overvalued. 

To better understand the results, I created a scatter plot visualization using the forecasted values along the x-axis and the current valuation along the y-axis. The data points are the current valuations for each company, and the trendline is the regression forecast. I labeled several companies on both sides of the trendline to identify companies that could be considered good opportunities to buy or sell. This simple screening model is only a brief example of the great value that the Bloomberg Terminal and the Bloomberg Excel add-in can provide.

