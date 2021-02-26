---
title: "Bloomberg Automotive Analysis"
date: 2020-02-28
tags: [data analysis, data science, linear regression]
header:
  image: "/images/automobile-production.png" 
excerpt: "Linear regression determining whether auto manufacturers and suppliers are overvalued or undervalued. _Tools: Bloomberg Terminal, MS Excel_"
mathjax: "true"
---
- Click the following link to download the [MS Excel file](https://github.com/mdreck/mdreck.github.io/blob/master/bloomberg_analysis/bloomberg_auto_analysis.xlsm) or view the PDF [here](https://github.com/mdreck/mdreck.github.io/blob/master/bloomberg_analysis/bloomberg_auto_analysis.pdf)

  _Tools: Bloomberg Terminal, MS Excel_

### Bloomberg Automotive Analysis
 The Bloomberg automotive analysis above, performs a regression on automotive industry data to determine whether auto suppliers and manufacturers are being undervalued or overvalued. The data was downloaded using a Bloomberg Terminal.
 
In Bloomberg we'll use the ‘RAY 3000 Index’ command to pull up the Russell 3000 Index and deleted all columns except the Ticker and Name columns. We add columns by typing GICS in the fields search bar and added the following: Sector Name, Industry Name, Industry Group Name, Sub-Industry Name. The Russell 3000 index data was then exported from Bloomberg into excel (See 'Russell 3000' worksheet).
In excel, sort the Russell 3000 Index by GICS Sub-Industry Name and performed the following: 
- Remove all companies (rows) that did not include the Sub-Industry ‘Auto Parts and Equipment’ or ‘Automobile Manufacturers’
- Remove all columns other than the ‘Ticker’ and ‘Name’ columns and sort data A to Z

Use the Bloomberg excel add-in to add fundamental financial data related to growth, profitability and risk, and compared those factors to a valuation metric (The Bloomberg Excel add-in is only available when using a Bloomberg machine). 
Use ‘Find Fields’ under the Bloomberg add-in tab to add financial data. We'll select the following fields for our analysis:
- Independent Variables:
  - ‘SALES_3YR_AVG_GROWTH’ and ‘GROSS_MARGIN’ to measure of growth and profitability 
  - ‘TOT_DEBT_TO_EBITDA’ (Debt / EBITDA) is included as a factor of risk and measures a company’s debt exposure. EBITDA is a company’s earnings before interest, tax, depreciation, and amortization
- Dependent Variables:
  - ‘EV_TO_T12M_EBITDA’ (Enterprise Value / EBITDA) is a valuation metric that represents the fair market value of a company. Enterprise value = mkt cap + preferred shares + minority interest + debt – total cash

Once the column fields are populated, use the BDP function to pull data for each company. The functions are viewable on the hidden ‘Bloomberg Data’ sheet within the excel workbook (the values will not populate unless you are on a Bloomberg Terminal, so I've copied and pasted that data as values in the ‘Model’ sheet).

We’ve selected these financial indicators because one would expect companies with high sales growth, high gross margins, and low debt to have a higher valuation than companies with poor sales growth and gross margins, and high debt. We'll perform a regression to evaluate the performance of the stock screening model and determine if the independent variables are good predictors of the market valuation (dependent variable). Execute the regression using the Data Analysis ToolPak in Excel. Use the market valuation (EV_TO12M_EBITDA) as the ‘Input Y Range’ and the independent variables from above as the ‘Input X Range’ with a confidence level of 95%. The null hypothesis that we are testing is that the changes to the independent variables are not associated with changes to the dependent variable. Ideally, we'd like to reject this null hypothesis in favor of a non-zero correlation between the independent variables and the dependent variable. Such a relationship would give us the ability to use this model to identify stocks that are undervalued or overvalued, but more on that later.  

The regression output is included in the ‘Model’ sheet. Located under the Regression Statistics table are the ‘goodness of fit’ measures. The most important of them for our purposes is the adjusted R square statistic. It determines the predictive value of the independent variables on the dependent variable. This indicates that 36% of the variation in the dependent variable can be explained by the independent variables. In other words, 36% of the values fit the model, which is not a particularly strong relationship but at least it’s positive. The Multiple R statistic is the correlation coefficient and describes the linear relationship is, so there is a positive correlation between the change in the independent variables and the dependent variable. 
 
To determine whether changes in the independent variables effect the dependent variable, we will evaluate their p-values. To reject the null hypothesis and identify the independent variable as statistically significant, the independent variable's p-value must be less than our significance level of 5% (0.05). Looking at the Coefficients table, we see that the p-value of DEBT to EBITDA is well above 5% and fails to reject the null hypothesis and is not a good predictor of market valuation. In future iterations of this screening model we would remove DEBT to EBITDA or replace it in search of another statistically significant variable. Sales growth and gross margin are statistically significant because of their low p-values and can be considered worthwhile for the purposes of our analysis. 

The coefficients column provides the slope of the variable. Sales growth and gross margin are both positive slopes just as we would expect that companies with good growth and high profits would have a higher valuation. For every one-unit increase in sales growth and gross margin, market valuation increases by an average of 0.16, and 0.39 cents respectively. DEBT to EBITDA has a positive slope of 0.26, however, it failed to reject the null hypothesis above and so its coefficient is equal to zero.   

Now that we have the intercept and slopes, we can forecast what the company should be valued at according to the linear regression equation (y = mx + b). The values and function are displayed in the ‘Model’ sheet under the Forecast EV to EBITDA column. From the forecast, we can identify companies that are overvalued or undervalued compared to their current valuation. Companies with forecasts that are higher than their current valuation should be considered undervalued, whereas companies with a lower forecast than their current valuatoin should be considered overvalued. 

To better understand the results, we'll create a scatter plot visualization using the forecasted values along the x-axis and the current valuation along the y-axis. The data points are the current valuations for each company, and the trendline is the regression forecast. I've labeled several companies on both sides of the trendline to identify companies that could be considered good opportunities to buy or sell. This simple model is only a brief example of the powerful insight that the Bloomberg Terminal and Excel add-in can provide.
