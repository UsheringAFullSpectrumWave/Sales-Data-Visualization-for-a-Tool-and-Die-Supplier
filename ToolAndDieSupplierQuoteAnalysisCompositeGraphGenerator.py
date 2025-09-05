import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, \
    HoverTool, Whisker, Panel, Tabs
from bokeh.models.widgets import RadioButtonGroup, Div, RangeSlider, DateRangeSlider, DataTable, TableColumn, \
    DateFormatter, Button
#from bokeh.models.glyphs import Text
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import column, row, widgetbox, Spacer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #r2_score
import statistics
import time
import math

def convertingDatesToOrdinals(dates, ordinalList):
    for index, x in enumerate(dates):
        y = x
        ordinalList.append(y.toordinal())

    return dates          

def computingMeansAndStdDeviation(plotData, monthYearArray):
    dictDataBeforeProcessing = {}
    
    for mY in monthYearArray:
        #key value list positions determine [['Unit Margin], 'Num Quotes']
        dictDataBeforeProcessing[mY] = [[], 0]
    
    for index, date in enumerate(plotData['Date']):
        dateProcessed = date.strftime('%B %Y')
        unit_margin = plotData['Unit Margin'][index]
        dictDataBeforeProcessing[dateProcessed][0].append(unit_margin)
        dictDataBeforeProcessing[dateProcessed][1] += 1
    
    dictData = {'Avg Margin': [],
                'Std Dev': [],
                'Num Quotes': [],
                'Date': []
                }
    
    for my in monthYearArray:
        unitMarginsList = dictDataBeforeProcessing[my][0]
        numQuotes = dictDataBeforeProcessing[my][1]
        lengthUML = len(unitMarginsList)
        
        if lengthUML > 1:
            avg = statistics.mean(unitMarginsList)
            stdDev = statistics.stdev(unitMarginsList)
        elif lengthUML == 1:
            avg = statistics.mean(unitMarginsList)
            stdDev = 0
        else:
            avg = 0
            stdDev = 0
        
        dictData['Avg Margin'].append(avg)
        dictData['Std Dev'].append(stdDev)
        dictData['Num Quotes'].append(numQuotes)
        
    dictData['Date'].extend(monthYearArray)
        
    return dictData


def computingSumsPerMonthYear(plotData, monthYearList, monthlyUniqueItemsTotal):
    #list key value [Sum Canadian Dollars, Item Quantity, Amount of Unique Items]
    dictData = {}
    for mY in monthYearList:
        dictData[mY] = [0, 0, 0]
    
    if len(plotData['Date']) == 0:
        return (dictData, monthlyUniqueItemsTotal)
    else:
        tempMonthYear = plotData['Date'][0].strftime('%B %Y')
    itemListPerMonth = []
    uniqueItemListPerMonth = []
    length = len(plotData['Date'])
    for index, date in enumerate(plotData['Date']):
        dateProcessed = date.strftime('%B %Y')
        dictData[dateProcessed][0] += plotData['Extended Price'][index]
        dictData[dateProcessed][1] += plotData['Quantity'][index]
        if dateProcessed == tempMonthYear and index != length - 1:
            itemListPerMonth.append(plotData['Item Code'][index])
            monthlyUniqueItemsTotal[dateProcessed].append(plotData['Item Code'][index])
        else:
            uniqueItemListPerMonth = list(set(itemListPerMonth))
            dictData[dateProcessed][2] += len(uniqueItemListPerMonth)
            itemListPerMonth = []
            
        tempMonthYear = dateProcessed
        
    return (dictData, monthlyUniqueItemsTotal)
 
    
def processedQuoteDataForDetailSales(dictForProcessing):
    returnedDictionary = {'Extended Price Sum': [],
                          'Date': []
                          }
    
    nestedListReturned = list(dictForProcessing.items())
    nestedListReturned.sort(key = lambda x: x[0])
    
    for x in nestedListReturned:
        returnedDictionary['Extended Price Sum'].append(x[1])
        returnedDictionary['Date'].append(x[0])
    
    return returnedDictionary

    
def quoteDataForDetailSales(detailSalesPlotData):
    dictData = {}
    
    for index, x in enumerate(detailSalesPlotData['Date']):
        price = detailSalesPlotData['Extended Price'][index]
        if x not in dictData:
            dictData[x] = price
        else:
            dictData[x] += price
        
    dictDataProcessed = processedQuoteDataForDetailSales(dictData)
    
    return dictDataProcessed


def determineMinAndMaxDates(winsDates, lossesDates):
    
    if winsDates == [] and lossesDates == []:
        minDateWins = 0
        maxDateWins = 1
        minDateLosses = 0
        maxDateLosses = 1
    elif winsDates == []:
        minDateWins = lossesDates[0]
        maxDateWins = lossesDates[-1]
        minDateLosses = lossesDates[0]
        maxDateLosses = lossesDates[-1]
    elif lossesDates == []:
        minDateWins = winsDates[0]
        maxDateWins = winsDates[-1]
        minDateLosses = winsDates[0]
        maxDateLosses = winsDates[-1]
    else:
        minDateWins = winsDates[0]
        maxDateWins = winsDates[-1]
        minDateLosses = lossesDates[0]
        maxDateLosses = lossesDates[-1]
    
    if minDateWins < minDateLosses:
        minDate = minDateWins
    else:
        minDate = minDateLosses
    
    if maxDateWins > maxDateLosses:
        maxDate = maxDateWins
    else:
        maxDate = maxDateLosses
        
    if maxDate == minDate:
        maxDate = maxDate + timedelta(days = 1)
        minDate = minDate - timedelta(days = 1)

    return (minDate, maxDate)


def linearRegressionComputationAndGlyph(datapointDatesWinsConverted, datapointWins,
                                        datapointDatesLossesConverted, 
                                        plt, datapointLosses,
                                        plotDataWins, plotDataLosses):
    #Wins Data Processing
    X_trainingWins = np.array(datapointDatesWinsConverted).reshape(-1, 1)
    X_testWins = np.array(datapointDatesWinsConverted).reshape(-1, 1)
    y_trainingWins = np.array(datapointWins)
    
    if len(X_trainingWins) > 0:
        regressionInfoWins = LinearRegression().fit(X_trainingWins, y_trainingWins)
        PredictWins = regressionInfoWins.predict(X_testWins)
        coefOfDeterminationWins = str(regressionInfoWins.score(X_trainingWins, y_trainingWins))
        MSE_RegressionWins = str(mean_squared_error(y_trainingWins, PredictWins))
    else:
        coefOfDeterminationWins = ''
        MSE_RegressionWins = ''
        PredictWins = None
    
    #Losses Data Processing
    X_trainingLosses = np.array(datapointDatesLossesConverted).reshape(-1, 1)
    X_testLosses = np.array(datapointDatesLossesConverted).reshape(-1, 1)
    y_trainingLosses = np.array(datapointLosses)
    
    if len(X_trainingLosses) > 0:
        regressionInfoLosses = LinearRegression().fit(X_trainingLosses, y_trainingLosses)
        PredictLosses = regressionInfoLosses.predict(X_testLosses)
        coefOfDeterminationLosses = str(regressionInfoLosses.score(X_trainingLosses, y_trainingLosses))
        MSE_RegressionLosses = str(mean_squared_error(y_trainingLosses, PredictLosses))

    else:
        coefOfDeterminationLosses = ''
        MSE_RegressionLosses = ''
        PredictLosses = None
    
    if PredictWins is not None:
        regSourceWins = ColumnDataSource(dict(x = plotDataWins['Date'], y = PredictWins))
    else:
        regSourceWins = None
    if PredictLosses is not None:
        regSourceLosses  = ColumnDataSource(dict(x = plotDataLosses['Date'], y = PredictLosses))
    else:
        regSourceLosses = None
    
    #Rendering
    if len(X_trainingWins) > 0 and PredictWins is not None:
        renderRegressionWins = plt.line(x = 'x', y = 'y', color = 'blue', 
                                    alpha = 0.8, line_width = 4, legend = 'Regression Line (Wins)',
                                    source = regSourceWins)
    else:
        renderRegressionWins = None
    
    if len(X_trainingLosses) > 0 and PredictLosses is not None:
        renderRegressionLosses = plt.line(x = 'x', y = 'y', 
                                          color = 'red', alpha = 0.8, line_width = 4, 
                                          legend = 'Regression Line (Losses)',
                                          source = regSourceLosses)
    else:
        renderRegressionLosses = None

    return (renderRegressionWins, coefOfDeterminationWins, MSE_RegressionWins,
            renderRegressionLosses, coefOfDeterminationLosses, MSE_RegressionLosses)


def linearRegressionInfoDivs(SP_YAxis_Flag, coefOfDeterminationWins, MSE_RegressionWins,
                             coefOfDeterminationLosses, MSE_RegressionLosses):
    if SP_YAxis_Flag == 0:
        TEXT_Info_One = 'UNIT MARGIN LINEAR REGRESSION INFORMATION'
    else:
        TEXT_Info_One = 'QUANTITY LINEAR REGRESSION INFORMATION'
        
    TEXT_Info_Two = '\nLinear Regression Wins'
    TEXT_Info_Three = 'Coefficient of Determination: ' + coefOfDeterminationWins
    TEXT_Info_Four = 'Mean Squared Error: ' + MSE_RegressionWins
    TEXT_Info_Five = 'Linear Regression Losses'
    TEXT_Info_Six = 'Coefficient of Determination: ' + coefOfDeterminationLosses
    TEXT_Info_Seven = 'Mean Squared Error: ' + MSE_RegressionLosses
    informationMetrics_One = Div(text = TEXT_Info_One)
    informationMetrics_Two = Div(text = TEXT_Info_Two)
    informationMetrics_Three = Div(text = TEXT_Info_Three)
    informationMetrics_Four = Div(text = TEXT_Info_Four)
    informationMetrics_Five = Div(text = TEXT_Info_Five)
    informationMetrics_Six = Div(text = TEXT_Info_Six)
    informationMetrics_Seven = Div(text = TEXT_Info_Seven)
    informationMetricsList = [informationMetrics_One, informationMetrics_Two, informationMetrics_Three,
                             informationMetrics_Four, informationMetrics_Five, informationMetrics_Six,
                             informationMetrics_Seven]
    
    return informationMetricsList


def infoMetricsLinearRegressionOrganization(informationMetricsList):
    infoMetricsCol = column(widgetbox(informationMetricsList[0]), widgetbox(informationMetricsList[1]), 
                            widgetbox(informationMetricsList[2]), widgetbox(informationMetricsList[3]), 
                            widgetbox(informationMetricsList[4]), widgetbox(informationMetricsList[5]), 
                            widgetbox(informationMetricsList[6]))
    
    return infoMetricsCol

  
def scatterPlotGraph(plotDataWins, plotDataLosses, alphabSortUniqueCompanyList, product_Groupings_Dict,
                     item_Master_List, companyProductGroupItem_Breakdown):
    #SCATTER PLOT 1: FOR MARGINS
    TITLE = 'Unit Margins Over Time For Items Won and Lost From Quotes'
    TOOLS = 'pan, box_select, lasso_select, wheel_zoom, reset, save'
    
    pltMargin_SP = figure(tools = TOOLS, toolbar_location = 'right', x_axis_type = 'datetime',
                  plot_width = 1400, title = TITLE, output_backend = 'webgl')
    pltMargin_SP.toolbar.logo = None
    pltMargin_SP.xaxis.axis_label = 'Date Quoted'
    pltMargin_SP.xaxis.formatter = DatetimeTickFormatter(seconds = '%m-%d-%Y',
                                                minutes = '%m-%d-%Y',
                                                hours = '%m-%d-%Y',
                                                days = '%m-%d-%Y',
                                                months = '%m-%d-%Y',
                                                years ='%m-%d-%Y')
    pltMargin_SP.yaxis.axis_label = 'Unit Margin (%)'
    pltMargin_SP.grid.grid_line_color = 'grey'
  
    dataSourceWins = ColumnDataSource(plotDataWins)
    dataSourceWins_Copy = ColumnDataSource(plotDataWins)
    renderScatterMarginsWins = pltMargin_SP.circle(x = 'Date', y = 'Unit Margin', size = 12, color = 'blue', 
                                   line_color = 'black', fill_alpha = 0.5, legend = 'Item(s) Won', 
                                   source = dataSourceWins)
    
    dataSourceLosses = ColumnDataSource(plotDataLosses)
    dataSourceLosses_Copy = ColumnDataSource(plotDataLosses)
    renderScatterMarginsLosses = pltMargin_SP.circle(x = 'Date', y = 'Unit Margin', size = 12, color = 'red', 
                                     line_color = 'black', fill_alpha = 0.5, legend = 'Item(s) Lost', 
                                     source = dataSourceLosses)
    
    datapointDatesWins = plotDataWins['Date']
    datapointDatesWins_TEST = []
    datapointDatesWins_TEST.extend(plotDataWins['Date'])
    datapointDatesWinsConverted = []
    datapointDatesWinsConverted_TEST = []
    convertingDatesToOrdinals(datapointDatesWins, datapointDatesWinsConverted)
    convertingDatesToOrdinals(datapointDatesWins_TEST, datapointDatesWinsConverted_TEST)
    
    datapointDatesLosses = plotDataLosses['Date']
    datapointDatesLosses_TEST = []
    datapointDatesLosses_TEST.extend(plotDataLosses['Date'])
    datapointDatesLossesConverted = []
    datapointDatesLossesConverted_TEST = []
    convertingDatesToOrdinals(datapointDatesLosses, datapointDatesLossesConverted)
    convertingDatesToOrdinals(datapointDatesLosses_TEST, datapointDatesLossesConverted_TEST)
    
    datapointMarginsWins = plotDataWins['Unit Margin']
    datapointMarginsLosses = plotDataLosses['Unit Margin']
    
    renderRegressionMarginsWins, coefOfDeterminationMarginsWins, MSE_RegressionMarginsWins, \
    renderRegressionMarginsLosses, coefOfDeterminationMarginsLosses, MSE_RegressionMarginsLosses = \
    linearRegressionComputationAndGlyph(datapointDatesWinsConverted, datapointMarginsWins,
                                        datapointDatesLossesConverted, 
                                        pltMargin_SP, datapointMarginsLosses,
                                        plotDataWins, plotDataLosses)


    hoverPoints = HoverTool(tooltips = [
                    ('Company', '@{Company}'),
                    ('Item Code', '@{Item Code}'),
                    ('Unit Margin', '@{Unit Margin}{0.00}%'),
                    ('Quantity', '@Quantity'),
                    ('Date', '@Date{%m-%d-%Y}')], 
                    formatters = {'Date': 'datetime'},
                    renderers = [renderScatterMarginsWins, renderScatterMarginsLosses])
        
    hoverLine = HoverTool(tooltips = [
                    ('Unit Margin', '@y{0.00}%')],
                    renderers = [renderRegressionMarginsWins, renderRegressionMarginsLosses])
    
    pltMargin_SP.add_tools(hoverPoints)
    pltMargin_SP.add_tools(hoverLine)
    
    pltMargin_SP.legend.click_policy = 'hide'
    pltMargin_SP.legend.title = 'Interactive Unit Margins Legend'
    
    #SCATTER PLOT 2: FOR QUANTITIES
    TITLE = 'Quantities Over Time For Items Won and Lost From Quotes'
    pltQuantities_SP = figure(tools = TOOLS, toolbar_location = 'right', x_axis_type = 'datetime',
                  plot_width = 1400, title = TITLE, output_backend = 'webgl')
    pltQuantities_SP.toolbar.logo = None
    pltQuantities_SP.xaxis.axis_label = 'Date Quoted'
    pltQuantities_SP.xaxis.formatter = DatetimeTickFormatter(seconds = '%m-%d-%Y',
                                                minutes = '%m-%d-%Y',
                                                hours = '%m-%d-%Y',
                                                days = '%m-%d-%Y',
                                                months = '%m-%d-%Y',
                                                years ='%m-%d-%Y')
    pltQuantities_SP.yaxis.axis_label = 'Quantity'
    pltQuantities_SP.grid.grid_line_color = 'grey'

    renderScatterQuantitiesWins = pltQuantities_SP.circle(x = 'Date', y = 'Quantity', size = 12, color = 'blue', 
                                   line_color = 'black', fill_alpha = 0.5, legend = 'Item(s) Won', 
                                   source = dataSourceWins)

    renderScatterQuantitiesLosses = pltQuantities_SP.circle(x = 'Date', y = 'Quantity', size = 12, color = 'red', 
                                     line_color = 'black', fill_alpha = 0.5, legend = 'Item(s) Lost', 
                                     source = dataSourceLosses)

    datapointQuantitiesWins = plotDataWins['Quantity']
    datapointQuantitiesLosses = plotDataLosses['Quantity']

    renderRegressionQuantitiesWins, coefOfDeterminationQuantitiesWins, MSE_RegressionQuantitiesWins, \
    renderRegressionQuantitiesLosses, coefOfDeterminationQuantitiesLosses, MSE_RegressionQuantitiesLosses = \
    linearRegressionComputationAndGlyph(datapointDatesWinsConverted, datapointQuantitiesWins,
                                        datapointDatesLossesConverted, 
                                        pltQuantities_SP, datapointQuantitiesLosses,
                                        plotDataWins, plotDataLosses)

    hoverPointsQuantities = HoverTool(tooltips = [
                    ('Company', '@{Company}'),
                    ('Item Code', '@{Item Code}'),
                    ('Quantity', '@Quantity'),
                    ('Unit Margin', '@{Unit Margin}{0.00}%'),
                    ('Date', '@Date{%m-%d-%Y}')], 
                    formatters = {'Date': 'datetime'},
                    renderers = [renderScatterQuantitiesWins, renderScatterQuantitiesLosses])
        
    hoverLineQuantities = HoverTool(tooltips = [
                    ('Quantity', '@Quantity')],
                    renderers = [renderRegressionQuantitiesWins, renderRegressionQuantitiesLosses])
    
    pltQuantities_SP.add_tools(hoverPointsQuantities)
    pltQuantities_SP.add_tools(hoverLineQuantities)
    
    pltQuantities_SP.legend.click_policy = 'hide'
    pltQuantities_SP.legend.title = 'Interactive Quantities Legend'
    

    #Scatter Plot Common Controls and Features
    SP_YAxis_Flag = 0
    informationMetricsList_UnitMargin = linearRegressionInfoDivs(SP_YAxis_Flag, 
                                                                 coefOfDeterminationMarginsWins, 
                                                                 MSE_RegressionMarginsWins,
                                                                 coefOfDeterminationMarginsLosses, 
                                                                 MSE_RegressionMarginsLosses)
    
    SP_YAxis_Flag = 1
    informationMetricsList_Quantities = linearRegressionInfoDivs(SP_YAxis_Flag, 
                                                                 coefOfDeterminationQuantitiesWins, 
                                                                 MSE_RegressionQuantitiesWins,
                                                                 coefOfDeterminationQuantitiesLosses, 
                                                                 MSE_RegressionQuantitiesLosses)
    
    callback_ON_OFF = CustomJS(args = dict(regLineW = renderRegressionMarginsWins, 
                                    regLineL = renderRegressionMarginsLosses,
                                    CirW = renderScatterMarginsWins,
                                    CirL = renderScatterMarginsLosses,
                                    info_LR_M = informationMetricsList_UnitMargin,
                                    info_LR_Q = informationMetricsList_Quantities),
                        code = '''
                        function turnRegON(info_LR_M, info_LR_Q){
                            for (i = 0; i < 7; i += 1){
                                    info_LR_M[i].visible = true;
                                    info_LR_Q.visible = true;
                            }
                        }
                                
                        function turnRegOFF(info_LR_M, info_LR_Q){
                            for (i = 0; i < 7; i += 1){
                                    info_LR_M[i].visible = false;
                                    info_LR_Q.visible = false;
                            }
                        }
                            
                        console.log(cb_obj);
                        var radioValue = cb_obj.active;
                        console.log(radioValue);
                        var selection;
                        var range;
                        if (radioValue == 0){
                            turnRegON(info);
                            selection = 'Lin. Reg. Info. ON';
                        }
                        else if (radioValue == 1){
                            turnRegOFF(info);
                            selection = 'Lin. Reg. Info. OFF';
                        }
                        
                        console.log(selection);
                      ''')
        
    radio_Button_Group_LR = RadioButtonGroup(labels = ['Add Linear Regression Info.', 
                                                    'Remove Linear Regression Info.'], 
                                          active = 0, callback = callback_ON_OFF)
    
    totalUnitMarginEntries = []
    totalUnitMarginEntries.extend(plotDataWins['Unit Margin'])
    totalUnitMarginEntries.extend(plotDataLosses['Unit Margin'])
    minimum_for_start = min(totalUnitMarginEntries)
    maximum_for_start = max(totalUnitMarginEntries)
    
    if minimum_for_start == maximum_for_start:
        minimum_for_start -= 10
        maximum_for_start += 10
    
    range_SliderForMargins = RangeSlider(start = minimum_for_start, end = maximum_for_start,
                                         value = (minimum_for_start, maximum_for_start),
                                         step = 0.1, title = 'Minimum and Maximum Margin Limits') 
    
    minDate, maxDate = determineMinAndMaxDates(plotDataWins['Date'], plotDataLosses['Date'])
    
    scatterPlot_DateSlider =  DateRangeSlider(start = minDate, end = maxDate,
                                         value = (minDate, maxDate),
                                         step = 1, title = 'Minimum and Maximum Dates Observed')
    
    minimum_for_start_Q = min([min(plotDataWins['Quantity']), min(plotDataLosses['Quantity'])])
    maximum_for_start_Q = max([max(plotDataWins['Quantity']), max(plotDataLosses['Quantity'])])
    
    if minimum_for_start_Q == maximum_for_start_Q:
        minimum_for_start_Q -= 10
        maximum_for_start_Q += 10
    
    range_SliderQuantities = RangeSlider(start = minimum_for_start_Q, end = maximum_for_start_Q,
                                         value = (minimum_for_start_Q, maximum_for_start_Q),
                                         step = 1, title = 'Minimum and Maximum Quantities Limits') 
    
    sortedCompanySource = ColumnDataSource(dict(Company = alphabSortUniqueCompanyList))
    column_Companies = [TableColumn(field = 'Company', title = 'Company')]
    titleTable_CompanyList = Div(text = 'Company List')
    data_Table_Company_List_To_Choose_From = DataTable(source = sortedCompanySource, 
                                                       columns = column_Companies,
                                                       width = 100 , height = 200, editable = False,
                                                       selectable = True,
                                                       reorderable = False)
    
    companyAddRemoveSource = ColumnDataSource(dict(Company = ['ALL']))
    column_Companies_AddRemove = [TableColumn(field = 'Company', title = 'Company')]
    titleTable_CompanyList_AddRemove = Div(text = 'Companies Selected')
    data_Table_Companies_Selected = DataTable(source = companyAddRemoveSource, 
                                                       columns = column_Companies_AddRemove,
                                                       width = 100 , height = 200, editable = False,
                                                       selectable = True,
                                                       reorderable = False)
    
    productGroupsList = product_Groupings_Dict.keys()
    alphaSortedProductGroupList = sorted(productGroupsList)
    
    alphaSortedItemList = sorted(item_Master_List)
    alphaSortedItemList.insert(0, 'ALL')
    
    alphaSortedProductGroupList.insert(0, 'ALL')
    productGroupFoundationDict = {'Product Group': alphaSortedProductGroupList}
    sortedProductGroupSource = ColumnDataSource(productGroupFoundationDict)
    column_Product_Groups = [TableColumn(field = 'Product Group', title = 'Product Group')]
    titleTable_ProductGroupList = Div(text = 'Product Groups')
    data_Table_ProductGroups_To_Chose_From = DataTable(source = sortedProductGroupSource, 
                                                       columns = column_Product_Groups,
                                                       width = 100 , height = 200, editable = False,
                                                       selectable = True,
                                                       reorderable = False)
    
    productGroupAddRemoveFoundationDict = {'Product Group': ['ALL']}
    productGroupAddRemoveSource = ColumnDataSource(productGroupAddRemoveFoundationDict)
    column_Product_Groups_AddRemove = [TableColumn(field = 'Product Group', title = 'Product Group')]
    titleTable_ProductGroupList_AddRemove = Div(text = 'Product Groups Selected')
    data_Table_Product_Groups_Selected = DataTable(source = productGroupAddRemoveSource, 
                                                       columns = column_Product_Groups_AddRemove,
                                                       width = 100 , height = 200, editable = False,
                                                       selectable = True,
                                                       reorderable = False)
    
    itemFoundationDict = {'Item Code': alphaSortedItemList}
    sortedItemCodeSource = ColumnDataSource(itemFoundationDict)
    column_Items = [TableColumn(field = 'Item Code', title = 'Item Code')]
    titleTable_ItemCodeList = Div(text = 'Item Codes')
    data_Table_ItemCodes_To_Chose_From = DataTable(source = sortedItemCodeSource, 
                                                       columns = column_Items,
                                                       width = 100 , height = 200, editable = False,
                                                       selectable = True,
                                                       reorderable = False)
    
    itemCodeAddRemoveFoundationDict = {'Item Code': ['ALL']}
    itemCodeAddRemoveSource = ColumnDataSource(itemCodeAddRemoveFoundationDict)
    column_Items_AddRemove = [TableColumn(field = 'Item Code', title = 'Item Code')]
    titleTable_Item_Codes_Add_Remove = Div(text = 'Item Codes Selected')
    data_Table_Item_Codes_Selected = DataTable(source = itemCodeAddRemoveSource, 
                                                       columns = column_Items_AddRemove,
                                                       width = 100 , height = 200, editable = False,
                                                       selectable = True,
                                                       reorderable = False)
    
    button_Add_Company = Button(label = 'Add Company', button_type = 'default')
    button_Remove_Company = Button(label = 'Remove Company', button_type = 'default')
    
    button_Add_Product_Group = Button(label = 'Add Product Group', button_type = 'default')
    button_Remove_Product_Group = Button(label = 'Remove Product Group', button_type = 'default') 
    
    button_Add_Item = Button(label = 'Add Item', button_type = 'default')
    button_Remove_Item = Button(label = 'Remove Item', button_type = 'default') 
    
    original_Scatter_Plot_Data = [dataSourceWins_Copy, dataSourceLosses_Copy]
    
    callback_Selection_SP = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,  
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = alphabSortUniqueCompanyList;
                        }
                        
                        var productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];
                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();
                        ''')
    
    callback_Selection_SP_ADD_Company = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    sortedCompanySource = sortedCompanySource,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,                              
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    product_Groupings_Dict = product_Groupings_Dict,
                                    item_Master_List = item_Master_List,
                                    alphaSortedItemList = alphaSortedItemList,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        var selected_Companies_Indices = sortedCompanySource.selected.indices;
                        var selected_Companies = [];
                        var add_Remove_Company_List = [];
                        
                        console.log(1);
                        var length;
                        try {
                                length = selected_Companies_Indices.length;
                        }
                        catch(err){
                                length = 0;
                        }
                        console.log(2);
                        
                        if (length == 0){
                                throw new Error('Early Purposeful Termination');
                        }
                        
                        
                        for (i = 0; i < length; i++){
                                selected_Companies.push(sortedCompanySource.data['Company'][selected_Companies_Indices[i]]);
                        }
                        console.log(3);
                        
                        if (selected_Companies.includes('ALL')){
                                companyAddRemoveSource.data = {
                                            'Company': ['ALL']
                                        };                        
                        }
                        console.log(4);
                        
                        if (companyAddRemoveSource.data['Company'][0] == 'ALL'){
                                add_Remove_Company_List = ['ALL'];
                            }
                        else{
                                add_Remove_Company_List = companyAddRemoveSource.data['Company'];
                                add_Remove_Company_List = add_Remove_Company_List.concat(selected_Companies);
                                add_Remove_Company_List = new Set(add_Remove_Company_List);
                                add_Remove_Company_List = Array.from(add_Remove_Company_List);
                                add_Remove_Company_List.sort();
                        }
                        console.log(5);
                        
                        companyAddRemoveSource.data = {
                                'Company': add_Remove_Company_List
                        };
                        console.log(6);
                        
                        companyAddRemoveSource.change.emit();
                          
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        var tempListProductGroups;
                        var productGroupList;
                        var productGroupsPresent = [];
                        var productGroupMatchList;
                        var tempListItems;
                        var itemList;
                        var itemCodesPresent = [];
                        var itemMatchList;
                        
                        var CML_length;
                        var PGM_length;
                        try{
                                CML_length = companyMatchList.length;
                        }
                        catch(err){
                                CML_length = 0;
                        }
                        console.log(7);
                        
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = alphabSortUniqueCompanyList;
                                productGroupList = Object.keys(product_Groupings_Dict);
                                itemList = alphaSortedItemList;
                                PGM_length = productGroupList.length;
                        }
                        else{
                                for (i = 0; i < CML_length; i++){
                                        tempListProductGroups = Object.keys(companyProductGroupItem_Breakdown[companyMatchList[i]]);
                                        productGroupsPresent = productGroupsPresent.concat(tempListProductGroups);
                                }
                                console.log(productGroupsPresent);
                                productGroupList = Array.from(new Set(productGroupsPresent)); 
                                productGroupList = productGroupList.sort();
                                
                                try{
                                    PGM_length = productGroupList.length;
                                }
                                catch(err){
                                    PGM_length = 0;
                                }
                                
                                for (i = 0; i < CML_length; i++){
                                    for (j = 0; j < PGM_length; j++){
                                            tempListItems = companyProductGroupItem_Breakdown[companyMatchList[i]][productGroupList[j]];
                                            itemCodesPresent = itemCodesPresent.concat(tempListItems);
                                    }
                                }
                                itemList = Array.from(new Set(itemCodesPresent));
                        }
                        console.log(8);
                        
                        try{
                                itemList_Length = itemList.length;
                        }
                        catch(err){
                                itemList_Length = 0;
                        }
                        console.log(9);
                        
                        if (PGM_length > 0){
                                productGroupList.unshift('ALL');
                        }
                        
                        if (itemList == alphaSortedItemList){
                        }
                        else if (itemList_Length > 0){
                                itemList.unshift('ALL');
                        }
                        console.log(10);
                        
                        sortedProductGroupSource.data = {
                                                    'Product Group': productGroupList
                                                    };
                        
                        sortedItemCodeSource.data = {
                                                'Item Code': itemList
                                                };
                        
                        sortedProductGroupSource.change.emit();
                        sortedItemCodeSource.change.emit();
                        
                        var productGroupAddRemove_Dynamic_Default = productGroupAddRemoveSource.data['Product Group'];
                        var productGroup_Intersection;
                        productGroup_Intersection = productGroupAddRemove_Dynamic_Default.filter(value => productGroupList.includes(value));
                        
                        console.log('PRODUCT GROUP INTERSECTION')
                        console.log(productGroup_Intersection)
                        
                        var itemAddRemove_Dynamic_Default = itemCodeAddRemoveSource.data['Item Code'];
                        var item_Intersection;
                        item_Intersection = itemAddRemove_Dynamic_Default.filter(value => itemList.includes(value));
                        
                        console.log('ITEMS INTERSECTION')
                        console.log(item_Intersection)
                        
                        var sorted_productGroup_Intersection = productGroup_Intersection.sort();
                        if (typeof sorted_productGroup_Intersection == 'undefined'){
                                sorted_productGroup_Intersection = [];
                        }
                        console.log(11);
                        
                        var sorted_item_Intersection = item_Intersection.sort();
                        if (typeof sorted_item_Intersection == 'undefined'){
                                sorted_item_Intersection = [];
                        }
                        console.log(12);
                        
                        productGroupAddRemoveSource.data = {
                                'Product Group': sorted_productGroup_Intersection
                                };
                        itemCodeAddRemoveSource.data = {
                                'Item Code': sorted_item_Intersection
                                };
                        
                        productGroupAddRemoveSource.change.emit();
                        itemCodeAddRemoveSource.change.emit();
                        
                        companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = sortedCompanySource.data['Company'].slice(1);
                        }
                        
                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        console.log(13);
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];

                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        console.log(14);
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();
                        ''')
                        
    callback_Selection_SP_REMOVE_Company = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    sortedCompanySource = sortedCompanySource,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,                              
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    product_Groupings_Dict = product_Groupings_Dict,
                                    item_Master_List = item_Master_List,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        var selected_Remove_Companies_Indices;
                        var selected_Remove_Companies = [];
                        
                        selected_Remove_Companies_Indices = companyAddRemoveSource.selected.indices;
                        
                        try {
                                length = selected_Remove_Companies_Indices.length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        if (length == 0){
                                throw new Error('Early Purposeful Termination');
                        }
                        
                        for (i = 0; i < length; i++){
                                selected_Remove_Companies.push(companyAddRemoveSource.data['Company'][selected_Remove_Companies_Indices[i]]);
                        }

                        try {
                                length = companyAddRemoveSource.data['Company'].length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        var resultant_Company_List = []
                        
                        for (i = 0; i < length; i++){
                            if (selected_Remove_Companies.includes(companyAddRemoveSource.data['Company'][i])){
                            }
                            else{
                                resultant_Company_List.push(companyAddRemoveSource.data['Company'][i])    
                            }
                        }
                        
                        companyAddRemoveSource.data = {
                            'Company': resultant_Company_List
                        };
                        
                        companyAddRemoveSource.change.emit();
                        
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        var tempListProductGroups;
                        var productGroupList;
                        var productGroupsPresent = [];
                        var productGroupMatchList;
                        var tempListItems;
                        var itemList;
                        var itemCodesPresent = [];
                        
                        var CML_length;
                        var PGM_length;
                        try{
                                CML_length = companyMatchList.length;
                        }
                        catch(err){
                                CML_length = 0;
                        }

                        console.log('Updating');
                        
                        for (i = 0; i < CML_length; i++){
                                tempListProductGroups = Object.keys(companyProductGroupItem_Breakdown[companyMatchList[i]]);
                                productGroupsPresent = productGroupsPresent.concat(tempListProductGroups);
                        }
                        console.log(productGroupsPresent);
                        productGroupList = Array.from(new Set(productGroupsPresent)); 
                        productGroupList = productGroupList.sort();
                                
                        try{
                                PGM_length = productGroupList.length;
                        }
                        catch(err){
                                PGM_length = 0;
                        }
                                
                        for (i = 0; i < CML_length; i++){
                                for (j = 0; j < PGM_length; j++){
                                        tempListItems = companyProductGroupItem_Breakdown[companyMatchList[i]][productGroupList[j]];
                                        itemCodesPresent = itemCodesPresent.concat(tempListItems);
                                }
                        }
                                
                        
                        itemList = Array.from(new Set(itemCodesPresent));
                        
                        var itemList_Length;
                        try{
                                itemList_Length = itemList.length;    
                        }
                        catch(err){
                                itemList_Length = 0;
                        }
                    
                    
                        if (PGM_length > 0){
                                productGroupList.unshift('ALL');
                        }
                        if (itemList_Length > 0){
                                itemList.unshift('ALL');
                        }
                    
                        sortedProductGroupSource.data = {
                                                    'Product Group': productGroupList
                                                    };
                        
                        sortedItemCodeSource.data = {
                                                'Item Code': itemList
                                                };
                        
                        sortedProductGroupSource.change.emit();
                        sortedItemCodeSource.change.emit();
                        
                        var productGroupAddRemove_Dynamic_Default = productGroupAddRemoveSource.data['Product Group'];
                        var productGroup_Intersection;
                        productGroup_Intersection = productGroupAddRemove_Dynamic_Default.filter(value => productGroupList.includes(value));
                        
                        console.log('PRODUCT GROUP INTERSECTION')
                        console.log(productGroup_Intersection)
                        
                        var itemAddRemove_Dynamic_Default = itemCodeAddRemoveSource.data['Item Code'];
                        var item_Intersection;
                        item_Intersection = itemAddRemove_Dynamic_Default.filter(value => itemList.includes(value));
                        
                        console.log('ITEMS INTERSECTION')
                        console.log(item_Intersection)
                        
                        var sorted_productGroup_Intersection = productGroup_Intersection.sort();
                        if (typeof sorted_productGroup_Intersection == 'undefined'){
                                sorted_productGroup_Intersection = []
                        }
                        
                        var sorted_item_Intersection = item_Intersection.sort();
                        if (typeof sorted_item_Intersection == 'undefined'){
                                sorted_item_Intersection = []
                        }
                        
                        productGroupAddRemoveSource.data = {
                                'Product Group': sorted_productGroup_Intersection
                                };
                        itemCodeAddRemoveSource.data = {
                                'Item Code': item_Intersection.sort()
                                };
                        
                        productGroupAddRemoveSource.change.emit();
                        itemCodeAddRemoveSource.change.emit();
                        
                        companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = sortedCompanySource.data['Company'].slice(1);
                        }
                        
                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];
                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();
                        ''')
                        
    callback_Selection_SP_ADD_ProductGroup = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    sortedCompanySource = sortedCompanySource,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,                              
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    product_Groupings_Dict = product_Groupings_Dict,
                                    item_Master_List = item_Master_List,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = alphabSortUniqueCompanyList.slice(1);
                        }
                        
                        var selected_ProductGroup_Indices = sortedProductGroupSource.selected.indices;
                        var selected_ProductGroup = [];
                        var add_Remove_ProductGroup_List = [];
                        
                        var length;
                        try {
                                length = selected_ProductGroup_Indices.length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        if (length == 0){
                                throw new Error('Early Purposeful Termination');
                        }
                        
                        
                        for (i = 0; i < length; i++){
                                selected_ProductGroup.push(sortedProductGroupSource.data['Product Group'][selected_ProductGroup_Indices[i]]);
                        }
                        
                        if (selected_ProductGroup.includes('ALL')){
                                productGroupAddRemoveSource.data = {
                                            'Product Group': ['ALL']
                                        };                        
                        }
                        
                        if (productGroupAddRemoveSource.data['Product Group'][0] == 'ALL'){
                                add_Remove_ProductGroup_List = ['ALL']; 
                            }
                        else{
                                add_Remove_ProductGroup_List = productGroupAddRemoveSource.data['Product Group'];
                                add_Remove_ProductGroup_List = add_Remove_ProductGroup_List.concat(selected_ProductGroup);
                                add_Remove_ProductGroup_List = new Set(add_Remove_ProductGroup_List);
                                add_Remove_ProductGroup_List = Array.from(add_Remove_ProductGroup_List);
                                add_Remove_ProductGroup_List.sort();
                        }
                        
                        productGroupAddRemoveSource.data = {
                                'Product Group': add_Remove_ProductGroup_List
                        };
                        
                        productGroupAddRemoveSource.change.emit()
                        
                        var productGroupMatchList = [];
                        var tempListItems;
                        var itemCodesPresent = [];

                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        else{
                                if (companyMatchList[0] == 'ALL'){
                                        productGroupMatchList = Object.keys(product_Groupings_Dict);
                                        productGroupMatchList.unshift('ALL')
                                }
                                else if (companyMatchList[0] == ''){
                                        productGroupMatchList = [];
                                }
                                else{
                                        for (i = 0; i < companyMatchList.length; i++){
                                                productGroupMatchList.concat(Object.keys(companyProductGroupItem_Breakdown[companyMatchList[i]]));
                                        }
                                        productGroupMatchList = Array.from(new Set(productGroupMatchList));
                                }
                        }
                        
                        var CML_length;
                        var PGM_length;
                        try{
                                CML_length = companyMatchList.length;
                        }
                        catch(err){
                                CML_length = 0;
                        }
                        try{
                                PGM_length = productGroupMatchList.length;
                        }
                        catch(err){
                                PGM_length = 0;
                        }
                        
                        console.log('CML_length');
                        console.log(CML_length);
                        console.log('PGM_length');
                        console.log(PGM_length);
                        
                        var itemList = [];
                        var ALL_Present_Product_Groups;
                        var All_Groups_Length;
                        
                        var index_I;
                        var index_J;
                        
                        if (productGroupAddRemoveSource.data['Product Group'][0] == 'ALL'){
                                index_J = 1;
                        }
                        else{
                                index_J = 0;
                        }
                        
                        if (companyAddRemoveSource.data['Company'][0] == 'ALL'){
                                index_I = 1;
                        }
                        else{
                                index_I = 0;
                        }
                        
                        for (i = index_I; i < CML_length; i++){
                                for (j = index_J; j < PGM_length; j++){
                                    tempListItems = companyProductGroupItem_Breakdown[companyMatchList[i]][productGroupMatchList[j]];
                                    itemCodesPresent = itemCodesPresent.concat(tempListItems);
                                }
                        }
                        itemList = Array.from(new Set(itemCodesPresent));
                        itemList = itemList.sort();
                        
                        console.log('itemList');
                        console.log(itemList);
                        
 
                        var itemList_Length;
                        try{
                                itemList_Length = itemList.length;    
                        }
                        catch(err){
                                itemList_Length = 0;
                        }
                        if (itemList_Length > 0){
                                itemList.unshift('ALL');
                        }

                        sortedItemCodeSource.data = {
                                                'Item Code': itemList
                                                };
                        
                        sortedItemCodeSource.change.emit();
                        
                        var itemAddRemove_Dynamic_Default = itemCodeAddRemoveSource.data['Item Code'];
                        var item_Intersection;
                        item_Intersection = itemAddRemove_Dynamic_Default.filter(value => itemList.includes(value));
                        
                        console.log('ITEMS INTERSECTION')
                        console.log(item_Intersection)
                        
                        var sorted_item_Intersection = item_Intersection.sort();
                        if (typeof sorted_item_Intersection == 'undefined'){
                                sorted_item_Intersection = [];
                        }
                        
                        itemCodeAddRemoveSource.data = {
                                'Item Code': sorted_item_Intersection
                                };
                        
                        itemCodeAddRemoveSource.change.emit();
                        
                        companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = sortedCompanySource.data['Company'].slice(1);
                        }
                        
                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];
                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();

                        ''')

    callback_Selection_SP_REMOVE_ProductGroup = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    sortedCompanySource = sortedCompanySource,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,                              
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    product_Groupings_Dict = product_Groupings_Dict,
                                    item_Master_List = item_Master_List,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        
                        var selected_Remove_ProductGroups_Indices;
                        var selected_Remove_ProductGroups = [];
                        
                        selected_Remove_ProductGroups_Indices = productGroupAddRemoveSource.selected.indices;
                        try {
                                length = selected_Remove_ProductGroups_Indices.length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        if (length == 0){
                                throw new Error('Early Purposeful Termination');
                        }
                        
                        for (i = 0; i < length; i++){
                                selected_Remove_ProductGroups.push(productGroupAddRemoveSource.data['Product Group'][selected_Remove_ProductGroups_Indices[i]]);
                        }

                        try {
                                length = productGroupAddRemoveSource.data['Product Group'].length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        var resultant_ProductGroups_List = []
                        
                        for (i = 0; i < length; i++){
                            if (selected_Remove_ProductGroups.includes(productGroupAddRemoveSource.data['Product Group'][i])){
                            }
                            else{
                                resultant_ProductGroups_List.push(productGroupAddRemoveSource.data['Product Group'][i])    
                            }
                        }
                        
                        productGroupAddRemoveSource.data = {
                            'Product Group': resultant_ProductGroups_List
                        };
                        
                        productGroupAddRemoveSource.change.emit();
                        
                        var productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        
                        var tempListItems;
                        var itemList;
                        var itemCodesPresent = [];
                        
                        var CML_length;
                        var PGM_length;
                        try{
                                CML_length = companyMatchList.length;
                        }
                        catch(err){
                                CML_length = 0;
                        }
                        try{
                                PGM_length = productGroupMatchList.length;
                        }
                        catch(err){
                                PGM_length = 0;
                        }
                        
                        var index_I;
                        var index_J;
                        
                        if (productGroupAddRemoveSource.data['Product Group'][0] == 'ALL'){
                                index_J = 1;
                        }
                        else{
                                index_J = 0;
                        }
                        
                        if (companyAddRemoveSource.data['Company'][0] == 'ALL'){
                                index_I = 1;
                        }
                        else{
                                index_I = 0;
                        }
                        
                        for (i = index_I; i < CML_length; i++){
                            for (j = index_J; j < PGM_length; j++){
                                    tempListItems = companyProductGroupItem_Breakdown[companyMatchList[i]][productGroupMatchList[j]];
                                    itemCodesPresent = itemCodesPresent.concat(tempListItems);
                                }
                        }
                        itemList = Array.from(new Set(itemCodesPresent));
                        itemList = itemList.sort();
                        
                        var itemList_Length;
                        try{
                                itemList_Length = itemList.length;    
                        }
                        catch(err){
                                itemList_Length = 0;
                        }
                        if (itemList_Length > 0){
                                itemList.unshift('ALL');
                        }

                        sortedItemCodeSource.data = {
                                                'Item Code': itemList
                                                };
                        
                        sortedItemCodeSource.change.emit();
                        
                        
                        var itemAddRemove_Dynamic_Default = itemCodeAddRemoveSource.data['Item Code'];
                        var item_Intersection;
                        item_Intersection = itemAddRemove_Dynamic_Default.filter(value => itemList.includes(value));
                        
                        console.log('ITEMS INTERSECTION')
                        console.log(item_Intersection)
                        
                        var sorted_item_Intersection = item_Intersection.sort();
                        if (typeof sorted_item_Intersection == 'undefined'){
                                sorted_item_Intersection = [];
                        }
                        
                        itemCodeAddRemoveSource.data = {
                                'Item Code': sorted_item_Intersection
                                };
                        
                        itemCodeAddRemoveSource.change.emit();
                        
                        companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = sortedCompanySource.data['Company'].slice(1);
                        }
                        
                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];
                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();

                        ''')

    callback_Selection_SP_ADD_Item = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    sortedCompanySource = sortedCompanySource,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,                              
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    product_Groupings_Dict = product_Groupings_Dict,
                                    item_Master_List = item_Master_List,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        
                        var selected_Item_Indices = sortedItemCodeSource.selected.indices;
                        var selected_Item = [];
                        var add_Remove_Item_List = [];
                        
                        var length;
                        try {
                                length = selected_Item_Indices.length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        if (length == 0){
                                throw new Error('Early Purposeful Termination');
                        }
                        
                        for (i = 0; i < length; i++){
                                selected_Item.push(sortedItemCodeSource.data['Item Code'][selected_Item_Indices[i]]);
                        }
                        
                        if (selected_Item.includes('ALL')){
                                itemCodeAddRemoveSource.data = {
                                            'Item Code': ['ALL']
                                        };                        
                        }
                        
                        
                        if (itemCodeAddRemoveSource.data['Item Code'][0] == 'ALL'){
                                add_Remove_Item_List = ['ALL'];
                            }
                        else{
                                add_Remove_Item_List = itemCodeAddRemoveSource.data['Item Code'];
                                add_Remove_Item_List = add_Remove_Item_List.concat(selected_Item);
                                add_Remove_Item_List = new Set(add_Remove_Item_List);
                                add_Remove_Item_List = Array.from(add_Remove_Item_List);
                                add_Remove_Item_List.sort();
                        }
                        
                        itemCodeAddRemoveSource.data = {
                                'Item Code': add_Remove_Item_List
                        };
                        
                        itemCodeAddRemoveSource.change.emit()
                        
                        companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = sortedCompanySource.data['Company'].slice(1);
                        }
                        
                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];
                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();

                        ''')
                        
    callback_Selection_SP_REMOVE_Item = CustomJS(args = dict(dataSourceWins = dataSourceWins, 
                                    dataSourceLosses = dataSourceLosses,
                                    original_Scatter_Plot_Data = original_Scatter_Plot_Data,
                                    range_SliderForMargins = range_SliderForMargins,
                                    scatterPlot_DateSlider = scatterPlot_DateSlider,
                                    sortedCompanySource = sortedCompanySource,
                                    alphabSortUniqueCompanyList = alphabSortUniqueCompanyList,
                                    companyAddRemoveSource = companyAddRemoveSource,
                                    sortedProductGroupSource = sortedProductGroupSource,
                                    productGroupAddRemoveSource = productGroupAddRemoveSource,
                                    sortedItemCodeSource = sortedItemCodeSource,                              
                                    itemCodeAddRemoveSource = itemCodeAddRemoveSource,
                                    product_Groupings_Dict = product_Groupings_Dict,
                                    item_Master_List = item_Master_List,
                                    companyProductGroupItem_Breakdown = companyProductGroupItem_Breakdown,
                                    range_SliderQuantities = range_SliderQuantities
                                    ),
                        code = '''
                        var original_Scatter_Plot_Data = original_Scatter_Plot_Data;
                        
                        var minValue = range_SliderForMargins.value[0];
                        var maxValue = range_SliderForMargins.value[1];
                        
                        var minDate = scatterPlot_DateSlider.value[0];
                        var maxDate = scatterPlot_DateSlider.value[1];
                        
                        var minQuantity = range_SliderQuantities.value[0];
                        var maxQuantity = range_SliderQuantities.value[1];
                        
                        const winsPointsLength = original_Scatter_Plot_Data[0].data['Unit Margin'].length;
                        const lossesPointsLength = original_Scatter_Plot_Data[1].data['Unit Margin'].length;
                        
                        var original_WINS_SOURCE = original_Scatter_Plot_Data[0];
                        var original_LOSSES_SOURCE = original_Scatter_Plot_Data[1];
                        
                        var companyMatchList = companyAddRemoveSource.data['Company'];
                        
                        var selected_Remove_Items_Indices;
                        var selected_Remove_Items = [];
                        
                        selected_Remove_Items_Indices = itemCodeAddRemoveSource.selected.indices;
                        try {
                                length = selected_Remove_Items_Indices.length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        if (length == 0){
                                throw new Error('Early Purposeful Termination');
                        }
                        
                        for (i = 0; i < length; i++){
                                selected_Remove_Items.push(itemCodeAddRemoveSource.data['Item Code'][selected_Remove_Items_Indices[i]]);
                        }

                        try {
                                length = itemCodeAddRemoveSource.data['Item Code'].length;
                        }
                        catch(err){
                                length = 0;
                        }
                        
                        var resultant_Items_List = []
                        
                        for (i = 0; i < length; i++){
                            if (selected_Remove_Items.includes(itemCodeAddRemoveSource.data['Item Code'][i])){
                            }
                            else{
                                resultant_Items_List.push(itemCodeAddRemoveSource.data['Item Code'][i])    
                            }
                        }
                        
                        itemCodeAddRemoveSource.data = {
                            'Item Code': resultant_Items_List
                        };
                        
                        itemCodeAddRemoveSource.change.emit();
                        
                        companyMatchList = companyAddRemoveSource.data['Company'];
                        if (companyMatchList[0] == 'ALL'){
                                companyMatchList = sortedCompanySource.data['Company'].slice(1);
                        }
                        
                        productGroupMatchList = productGroupAddRemoveSource.data['Product Group'];
                        if (productGroupMatchList[0] == 'ALL'){
                                productGroupMatchList = sortedProductGroupSource.data['Product Group'].slice(1);
                        }
                        
                        var itemMatchList = itemCodeAddRemoveSource.data['Item Code'];
                        if (itemMatchList[0] == 'ALL'){
                                itemMatchList = sortedItemCodeSource.data['Item Code'].slice(1);
                        }
                        
                        secondCopyDataSource_W = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        secondCopyDataSource_L = {'Company': [],
                                                  'Item Code': [],
                                                  'Unit Margin': [],
                                                  'Quantity': [],
                                                  'Date': [],
                                                  'Product Group': []
                                                  };
                        
                        console.log(companyMatchList);
                        console.log(productGroupMatchList);
                        console.log(itemMatchList);
                        
                        for (i = 0; i < winsPointsLength; i++){
                                if ((original_WINS_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_WINS_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_WINS_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_WINS_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_WINS_SOURCE.data['Date'][i] >= minDate) &&
                                (original_WINS_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_WINS_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_WINS_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_WINS_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_W['Company'].push(original_WINS_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_W['Item Code'].push(original_WINS_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_W['Unit Margin'].push(original_WINS_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_W['Quantity'].push(original_WINS_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_W['Date'].push(original_WINS_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_W['Product Group'].push(original_WINS_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesPointsLength; i++){
                                if ((original_LOSSES_SOURCE.data['Unit Margin'][i] >= minValue) &&
                                (original_LOSSES_SOURCE.data['Unit Margin'][i] <= maxValue) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] >= minQuantity) &&
                                (original_LOSSES_SOURCE.data['Quantity'][i] <= maxQuantity) &&
                                (original_LOSSES_SOURCE.data['Date'][i] >= minDate) &&
                                (original_LOSSES_SOURCE.data['Date'][i] <= maxDate) &&
                                (companyMatchList.includes(original_LOSSES_SOURCE.data['Company'][i])) &&
                                (productGroupMatchList.includes(original_LOSSES_SOURCE.data['Product Group'][i])) &&
                                (itemMatchList.includes(original_LOSSES_SOURCE.data['Item Code'][i]))){
                                        secondCopyDataSource_L['Company'].push(original_LOSSES_SOURCE.data['Company'][i]);
                                        secondCopyDataSource_L['Item Code'].push(original_LOSSES_SOURCE.data['Item Code'][i]);
                                        secondCopyDataSource_L['Unit Margin'].push(original_LOSSES_SOURCE.data['Unit Margin'][i]);
                                        secondCopyDataSource_L['Quantity'].push(original_LOSSES_SOURCE.data['Quantity'][i]);
                                        secondCopyDataSource_L['Date'].push(original_LOSSES_SOURCE.data['Date'][i]);
                                        secondCopyDataSource_L['Product Group'].push(original_LOSSES_SOURCE.data['Product Group'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                                                
                        dataSourceWins.data = {
                                'Company': secondCopyDataSource_W['Company'],
                                'Item Code': secondCopyDataSource_W['Item Code'],
                                'Unit Margin': secondCopyDataSource_W['Unit Margin'],
                                'Quantity': secondCopyDataSource_W['Quantity'],
                                'Date': secondCopyDataSource_W['Date'],
                                'Product Group': secondCopyDataSource_W['Product Group']
                                };
                        
                        dataSourceLosses.data = {
                                'Company': secondCopyDataSource_L['Company'],
                                'Item Code': secondCopyDataSource_L['Item Code'],
                                'Unit Margin': secondCopyDataSource_L['Unit Margin'],
                                'Quantity': secondCopyDataSource_L['Quantity'],
                                'Date': secondCopyDataSource_L['Date'],
                                'Product Group': secondCopyDataSource_L['Product Group']
                                };
                        
                        dataSourceWins.change.emit();
                        dataSourceLosses.change.emit();

                        ''')
                        
    
    range_SliderForMargins.js_on_change('value', callback_Selection_SP)
    scatterPlot_DateSlider.js_on_change('value', callback_Selection_SP)
    range_SliderQuantities.js_on_change('value', callback_Selection_SP)
    
    button_Add_Company.js_on_click(callback_Selection_SP_ADD_Company)
    button_Remove_Company.js_on_click(callback_Selection_SP_REMOVE_Company)
    button_Add_Product_Group.js_on_click(callback_Selection_SP_ADD_ProductGroup)
    button_Remove_Product_Group.js_on_click(callback_Selection_SP_REMOVE_ProductGroup)
    button_Add_Item.js_on_click(callback_Selection_SP_ADD_Item)
    button_Remove_Item.js_on_click(callback_Selection_SP_REMOVE_Item)
     
    columns_Data_Selected = [TableColumn(field = 'Company', title = 'Company'),
                            TableColumn(field = 'Item Code', title = 'Item Code'),
                            TableColumn(field = 'Unit Margin', title = 'Unit Margin'),
                            TableColumn(field = 'Quantity', title = 'Quantity'),
                            TableColumn(field = 'Date', title = 'Date', formatter = DateFormatter())]
    
    titleTable_W = Div(text = 'Selected Quote Data Won')
    data_Table_Selected_Data_Wins = DataTable(source = dataSourceWins, columns = columns_Data_Selected,
                                              width = 400 , height = 400, editable = False,
                                              selectable = False)
    
    titleTable_L = Div(text = 'Selected Quote Data Lost')
    data_Table_Selected_Data_Losses = DataTable(source = dataSourceLosses, columns = columns_Data_Selected,
                                                width = 400, height = 400, editable = False,
                                                selectable = False)
    
    spacer_Add_Remove_Buttons = Spacer(width = 50)
    spacer_Company_Control = Spacer(width = 100)
    add_Remove_Companies_Control = row(column(titleTable_CompanyList, data_Table_Company_List_To_Choose_From),
                                       spacer_Company_Control,
                                       column(titleTable_CompanyList_AddRemove, data_Table_Companies_Selected,
                                              button_Add_Company, spacer_Add_Remove_Buttons ,
                                              button_Remove_Company))
    
    add_Remove_Product_Groups_Control = row(column(titleTable_ProductGroupList, 
                                                   data_Table_ProductGroups_To_Chose_From),
                                       spacer_Company_Control,
                                           column(titleTable_ProductGroupList_AddRemove, 
                                                  data_Table_Product_Groups_Selected,
                                                  button_Add_Product_Group, spacer_Add_Remove_Buttons ,
                                                  button_Remove_Product_Group))
    
    add_Remove_Items_Control = row(column(titleTable_ItemCodeList, 
                                          data_Table_ItemCodes_To_Chose_From),
                                spacer_Company_Control,
                                    column(titleTable_Item_Codes_Add_Remove, 
                                           data_Table_Item_Codes_Selected,
                                           button_Add_Item, spacer_Add_Remove_Buttons ,
                                           button_Remove_Item))
    
    LR_UM_Divs = infoMetricsLinearRegressionOrganization(informationMetricsList_UnitMargin)
    LR_Q_Divs = infoMetricsLinearRegressionOrganization(informationMetricsList_Quantities)
    spacer_Between_LR_Info = Spacer(width = 100)
    
    spacer_Sliders = Spacer(width = 100)
    spacer_Tables = Spacer(width = 200)
    spacer_To_Controls = Spacer(width = 100)
    spacer_Between_Controls = Spacer(width = 100)
    spacer_Between_SPs = Spacer(width = 200)
    tab = Panel(child = column(pltMargin_SP, spacer_Between_SPs, pltQuantities_SP, 
                                row(widgetbox(range_SliderForMargins), spacer_Sliders,
                                    widgetbox(range_SliderQuantities), spacer_Sliders,
                                    widgetbox(scatterPlot_DateSlider)),
                                row(LR_UM_Divs, spacer_Between_LR_Info, LR_Q_Divs),  
                                widgetbox(radio_Button_Group_LR),
                                spacer_To_Controls, 
                                row(add_Remove_Companies_Control, spacer_Between_Controls,
                                    add_Remove_Product_Groups_Control, spacer_Between_Controls,
                                    add_Remove_Items_Control),
                                row(column(titleTable_W, data_Table_Selected_Data_Wins), spacer_Tables, 
                                    column(titleTable_L, data_Table_Selected_Data_Losses))),
                                title = 'Scatter Plots and Linear Regression')
    
    return tab
   
    
def marginMeansAndStdDeviationPlot(meanStdDevDataWins, meanStdDevDataLosses, monthYearArray):
    TITLE = 'Means And Standard Deviation Plot Over Time For Items Won and Lost From Quotes'
    TOOLS = 'pan, box_select, wheel_zoom, reset, save'
    
    plt = figure(tools = TOOLS, toolbar_location = 'right',
                  x_range = monthYearArray, plot_width = 1400, title = TITLE, output_backend = 'webgl')
    plt.toolbar.logo = None
    plt.xaxis.axis_label = 'Date'
    '''plt.xaxis.formatter = DatetimeTickFormatter(seconds = '%m-%Y',
                                                minutes = '%m-%Y',
                                                hours = '%m-%Y',
                                                days = '%m-%Y',
                                                months = '%m-%Y',
                                                years ='%m-%Y')'''
    plt.yaxis.axis_label = 'Avg Unit Margin (%) Per Month'
    plt.grid.grid_line_color = 'grey'
    
    winsAverages = meanStdDevDataWins['Avg Margin']
    winsStdDeviations = meanStdDevDataWins['Std Dev']
    winsNumQuotesPerAvg = meanStdDevDataWins['Num Quotes']
    winsDates = meanStdDevDataWins['Date']
    #winsDates = [x.strftime('%B %Y') for x in meanStdDevDataWins['Date']]
    #print(winsDates, '   winsDates\n') #TESTING
    lower = []
    upper = []
    for index, x in enumerate(winsAverages):
        lower.append(x - winsStdDeviations[index])
        upper.append(x + winsStdDeviations[index])
    sourceErrorWins = ColumnDataSource(dict(base = winsDates, lower = lower, upper = upper))
    sourceErrorWinsCopy = ColumnDataSource(dict(base = winsDates, lower = lower, upper = upper))
    sourceCircleWins = ColumnDataSource(dict(x = winsDates, y = winsAverages, z = winsStdDeviations,
                                             r = winsNumQuotesPerAvg))
    
    winsLine = plt.line(x = winsDates, y = winsAverages, color = 'blue', line_width = 2)
    winsPoint = plt.circle(x = 'x', y = 'y', color = 'blue', 
                           fill_color = 'white', size = 15, legend = 'Won',
                           source = sourceCircleWins)
    plt.add_layout(Whisker(source = sourceErrorWins, base = 'base', lower = 'lower', 
                           upper = 'upper'))
    
    lossesAverages = meanStdDevDataLosses['Avg Margin']
    lossesStdDeviations = meanStdDevDataLosses['Std Dev']
    lossesNumQuotesPerAvg = meanStdDevDataLosses['Num Quotes']
    lossesDates = meanStdDevDataLosses['Date']
    #lossesDates = [x.strftime('%B %Y') for x in meanStdDevDataLosses['Date']]
    #print(lossesDates, '   lossesDates\n') #TESTING
    lower = []
    upper = []
    for index, x in enumerate(lossesAverages):
        lower.append(x - lossesStdDeviations[index])
        upper.append(x + lossesStdDeviations[index])
    sourceErrorLosses = ColumnDataSource(dict(base = lossesDates, lower = lower, upper = upper))
    sourceErrorLossesCopy = ColumnDataSource(dict(base = lossesDates, lower = lower, upper = upper))
    sourceCircleLosses = ColumnDataSource(dict(x = lossesDates, y = lossesAverages, z = lossesStdDeviations,
                                               r = lossesNumQuotesPerAvg))
    

    lossesLine = plt.line(x = lossesDates, y = lossesAverages, color = 'red', line_width = 2)
    lossesPoint = plt.circle(x = 'x', y = 'y', color = 'red', 
                             fill_color = 'white', size = 15, legend = 'Lost',
                             source = sourceCircleLosses)
    plt.add_layout(Whisker(source = sourceErrorLosses, base = 'base', lower = 'lower', 
                           upper = 'upper'))
    
    hoverPoints = HoverTool(tooltips = [
                ('Avg Unit Margin', '@{y}{0.00}%'),
                ('Std. Dev.', '@{z}{0.00}%'),
                ('Num. Quote Lines', '@r'),
                ('Date', '@x')],
                renderers = [winsPoint, lossesPoint])
    plt.add_tools(hoverPoints)
    
    originalSourceError = [sourceErrorWinsCopy, sourceErrorLossesCopy]
    
    callback = CustomJS(args = dict(winsLine = winsLine, winsPoint = winsPoint, lossesLine = lossesLine,
                                    lossesPoint = lossesPoint, sourceErrorWins = sourceErrorWins, 
                                    sourceErrorLosses = sourceErrorLosses,
                                    originalSourceError = originalSourceError),
                        code = '''
                        console.log(cb_obj);
                        var radioValue = cb_obj.active;
                        console.log(radioValue);
                        var selection;
                        var dataErrorWins = sourceErrorWins.data;
                        var dataErrorLosses = sourceErrorLosses.data;
                        var sourceErrorOrginalW = originalSourceError[0];
                        var originalWins = sourceErrorOrginalW.data;
                        var sourceErrorOrginalL = originalSourceError[1];
                        var originalLosses = sourceErrorOrginalL.data;
                        if (radioValue == 0){
                            winsLine.visible = true;
                            winsPoint.visible = true;
                            sourceErrorWins.data['base'] = originalWins['base'];
                            sourceErrorWins.data['lower'] = originalWins['lower'];
                            sourceErrorWins.data['upper'] = originalWins['upper'];
                            selection = "Wins ON";
                            }
                        else if (radioValue == 1){
                            winsLine.visible = false;
                            winsPoint.visible = false;
                            sourceErrorWins.data['base'] = [];
                            sourceErrorWins.data['lower'] = [];
                            sourceErrorWins.data['upper'] = [];
                            selection = "Wins OFF";
                            }
                        else if (radioValue == 2){
                            lossesLine.visible = true;
                            lossesPoint.visible = true;
                            sourceErrorLosses.data['base'] = originalLosses['base'];
                            sourceErrorLosses.data['lower'] = originalLosses['lower'];
                            sourceErrorLosses.data['upper'] = originalLosses['upper'];
                            selection = "Losses ON";
                            }
                        else {
                            lossesLine.visible = false;
                            lossesPoint.visible = false;
                            sourceErrorLosses.data['base'] = [];
                            sourceErrorLosses.data['lower'] = [];
                            sourceErrorLosses.data['upper'] = [];
                            selection = "Losses OFF";
                            }
                        sourceErrorWins.change.emit();
                        sourceErrorLosses.change.emit();
                        console.log(selection)
                      ''')
    radio_Button_Group = RadioButtonGroup(labels = ['Add Wins', 'Remove Wins',
                                                    'Add Losses', 'Remove Losses'], 
                                        active = 0, callback = callback)
    
    return(plt, radio_Button_Group)


def salesLineGraph(detailSalesPlotDataWins, detailSalesPlotDataLosses, salesDataWins, 
                   salesDataLosses, monthYearArray, monthlyUniqueItemsTotal):
    TITLE_One = 'Sales Won And Lost Per Month From Quotes'
    TOOLS = 'pan, box_select, wheel_zoom, reset, save'
    
    pltSalesSum = figure(tools = TOOLS, toolbar_location = 'right',
                  x_range = monthYearArray, plot_width = 1400, title = TITLE_One, output_backend = 'webgl')
    pltSalesSum.toolbar.logo = None
    pltSalesSum.xaxis.axis_label = 'Date'
    pltSalesSum.yaxis.axis_label = 'Amount ($CAD)'
    pltSalesSum.grid.grid_line_color = 'grey'
    
    monthYearPresent = []
    #list key value [Sum Canadian Dollars, Item Quantity, Amount of Unique Items]
    salesDataWinsTransformed = {'Month Year': [], 'Sum': [], 
                                'Item Quantity': [], 'Unique Item Amount': []}
    for mY in monthYearArray:
        if mY in salesDataWins:
            monthYearPresent.append(mY)
            infoVector = salesDataWins[mY]
            salesDataWinsTransformed['Month Year'].append(mY)
            salesDataWinsTransformed['Sum'].append(infoVector[0])
            salesDataWinsTransformed['Item Quantity'].append(infoVector[1])
            salesDataWinsTransformed['Unique Item Amount'].append(infoVector[2])
    
    salesDataLossesTransformed = {'Month Year': [], 'Sum': [], 
                                'Item Quantity': [], 'Unique Item Amount': []}
    for mY in monthYearArray:
        if mY in salesDataLosses:
            monthYearPresent.append(mY)
            infoVector = salesDataLosses[mY]
            salesDataLossesTransformed['Month Year'].append(mY)
            salesDataLossesTransformed['Sum'].append(infoVector[0])
            salesDataLossesTransformed['Item Quantity'].append(infoVector[1])
            salesDataLossesTransformed['Unique Item Amount'].append(infoVector[2])
    
    monthYearPresent = list(set(monthYearPresent))
    monthYearPresent.sort(key = lambda x: datetime.strptime(x, '%B %Y'))
    totalSalesDataTransformed = {'Month Year': [], 'Sum': [], 
                                'Item Quantity': [], 'Unique Item Amount': []}
    for mY in monthYearPresent:
        if mY in salesDataWins and mY in salesDataLosses:
            infoVectorWins = salesDataWins[mY]
            infoVectorLosses = salesDataLosses[mY]
            infoVector = [x + y for x, y in zip(infoVectorWins, infoVectorLosses)]
        elif mY in salesDataWins:
            infoVector = salesDataWins[mY]
        elif mY in salesDataLosses:
            infoVector = salesDataLosses[mY]
        else:
            continue
        
        totalSalesDataTransformed['Month Year'].append(mY)
        totalSalesDataTransformed['Sum'].append(infoVector[0])
        totalSalesDataTransformed['Item Quantity'].append(infoVector[1])
        uniqueItemsTotalEntry = len(set(monthlyUniqueItemsTotal[mY]))
        totalSalesDataTransformed['Unique Item Amount'].append(uniqueItemsTotalEntry)
        
    sourceSalesPerMonth = ColumnDataSource(salesDataWinsTransformed)
    sourceLossesPerMonth = ColumnDataSource(salesDataLossesTransformed)
    sourceTotalCADPerMonth = ColumnDataSource(totalSalesDataTransformed)
        
    winsLinePerMonth = pltSalesSum.line(x = 'Month Year', y = 'Sum', color = 'blue', line_width = 2, 
                                        source = sourceSalesPerMonth)
    winsPointPerMonth = pltSalesSum.circle(x = 'Month Year', y = 'Sum', color = 'blue', 
                             fill_color = 'blue', size = 10, legend = 'Sales Won',
                             source = sourceSalesPerMonth)
    
    lossesLinePerMonth = pltSalesSum.line(x = 'Month Year', y = 'Sum', color = 'red', line_width = 2, 
                                          source = sourceLossesPerMonth)
    lossesPointPerMonth = pltSalesSum.circle(x = 'Month Year', y = 'Sum', color = 'red', 
                             fill_color = 'red', size = 10, legend = 'Sales Lost',
                             source = sourceLossesPerMonth)
    
    totalLinePerMonth = pltSalesSum.line(x = 'Month Year', y = 'Sum', color = 'orange', line_width = 2, 
                                         source = sourceTotalCADPerMonth)
    totalPointPerMonth = pltSalesSum.circle(x = 'Month Year', y = 'Sum', color = 'orange', 
                             fill_color = 'orange', size = 10, legend = 'Total Quoted',
                             source = sourceTotalCADPerMonth)
    
    hoverPoints = HoverTool(tooltips = [
                ('Amount (CAD)', '@Sum{1.11}'),
                ('Item Quantity', '@{Item Quantity}'),
                ('Number of Unique Items', '@{Unique Item Amount}'),
                ('Date', '@{Month Year}')],
                renderers = [winsPointPerMonth, lossesPointPerMonth,
                             totalPointPerMonth])
    
    pltSalesSum.add_tools(hoverPoints)
    
    callback_Monthly = CustomJS(args = dict(winsLinePerMonth = winsLinePerMonth, 
                                    winsPointPerMonth = winsPointPerMonth,
                                    lossesLinePerMonth = lossesLinePerMonth,
                                    lossesPointPerMonth = lossesPointPerMonth,
                                    totalLinePerMonth = totalLinePerMonth,
                                    totalPointPerMonth = totalPointPerMonth),
                        code = '''
                            console.log(cb_obj);
                            var radioValue = cb_obj.active;
                            console.log(radioValue);
                            var selection;
                            if (radioValue == 0){
                                    winsLinePerMonth.visible = true;
                                    winsPointPerMonth.visible = true;
                                    selection = 'Wins ON';
                            }
                            else if (radioValue == 1){
                                    winsLinePerMonth.visible = false;
                                    winsPointPerMonth.visible = false;
                                    selection = 'Wins OFF';
                            }
                            else if (radioValue == 2){
                                    lossesLinePerMonth.visible = true;
                                    lossesPointPerMonth.visible = true;
                                    selection = 'Loss ON';
                            }
                            else if (radioValue == 3){
                                    lossesLinePerMonth.visible = false;
                                    lossesPointPerMonth.visible = false;
                                    selection = 'Loss OFF';
                            }
                            else if (radioValue == 4){
                                    totalLinePerMonth.visible = true;
                                    totalPointPerMonth.visible = true;
                                    selection = 'Total ON';
                            }
                            else{
                                    totalLinePerMonth.visible = false;
                                    totalPointPerMonth.visible = false;
                                    selection = 'Total OFF';
                            }
                            console.log(selection);
                        ''')
    
    radio_Button_Group_MonthSum = RadioButtonGroup(labels = ['Add Wins', 'Remove Wins',
                                                    'Add Losses', 'Remove Losses',
                                                    'Add Total', 'Remove Total'], 
                                        active = 0, callback = callback_Monthly)
    
    #Plot two: THE DETAILED ONE
    TITLE_Two = 'Sales Won And Lost Per Quote'
    pltSalesDetail = figure(tools = TOOLS, toolbar_location = 'right',
                  x_axis_type = 'datetime', plot_width = 1400, title = TITLE_Two, 
                  output_backend = 'webgl')
    pltSalesDetail.toolbar.logo = None
    pltSalesDetail.xaxis.axis_label = 'Date'
    pltSalesDetail.yaxis.axis_label = 'Amount ($CAD)'
    pltSalesDetail.grid.grid_line_color = 'grey'
    
    
    detailSalesPlotDataTotal = {'Extended Price Sum': [],
                                'Date': []
                               }
    
    fullDataForAllDatesTotalCAD_Dictionary = {}
    for index, x in enumerate(detailSalesPlotDataWins['Date']):
        if x not in fullDataForAllDatesTotalCAD_Dictionary:
            fullDataForAllDatesTotalCAD_Dictionary[x] = detailSalesPlotDataWins['Extended Price Sum'][index] 
        else:
            fullDataForAllDatesTotalCAD_Dictionary[x] += detailSalesPlotDataWins['Extended Price Sum'][index]
            
    for index, x in enumerate(detailSalesPlotDataLosses['Date']):
        if x not in fullDataForAllDatesTotalCAD_Dictionary:
            fullDataForAllDatesTotalCAD_Dictionary[x] = detailSalesPlotDataLosses['Extended Price Sum'][index]
        else:
            fullDataForAllDatesTotalCAD_Dictionary[x] += detailSalesPlotDataLosses['Extended Price Sum'][index]
    
    fullDataForAllDatesTotalCADNestedList = list(fullDataForAllDatesTotalCAD_Dictionary.items())
    fullDataForAllDatesTotalCADNestedList.sort(key = lambda x: x[0])
    
    for x in fullDataForAllDatesTotalCADNestedList:
        detailSalesPlotDataTotal['Extended Price Sum'].append(x[1])
        detailSalesPlotDataTotal['Date'].append(x[0])
    
    sourceDetailSumsWins = ColumnDataSource(detailSalesPlotDataWins)
    sourceDetailSumsLosses = ColumnDataSource(detailSalesPlotDataLosses)
    sourceDetailSumsTotal = ColumnDataSource(detailSalesPlotDataTotal)
    
    sourceDetailSumsWins_Copy = ColumnDataSource(detailSalesPlotDataWins)
    sourceDetailSumsLosses_Copy = ColumnDataSource(detailSalesPlotDataLosses)
    sourceDetailSumsTotal_Copy = ColumnDataSource(detailSalesPlotDataTotal)
    
    detailedPriceWonLine = pltSalesDetail.line(x = 'Date', y = 'Extended Price Sum', color = 'blue', 
                                               line_width = 2, source = sourceDetailSumsWins,
                                               legend = 'Sales Won')
    detailedPriceLostLine = pltSalesDetail.line(x = 'Date', y = 'Extended Price Sum', color = 'red', 
                                               line_width = 2, source = sourceDetailSumsLosses,
                                               legend = 'Sales Lost')
    detailedPriceTotalLine = pltSalesDetail.line(x = 'Date', y = 'Extended Price Sum', color = 'orange', 
                                               line_width = 2, source = sourceDetailSumsTotal,
                                               legend = 'Total Quoted')
    
    hoverLine = HoverTool(tooltips = [
                ('Amount (CAD)', '@{Extended Price Sum}{1.11}'),
                ('Date', '@Date{%m-%d-%Y}')],
                formatters = {'Date': 'datetime'},
                renderers = [detailedPriceWonLine, detailedPriceLostLine,
                             detailedPriceTotalLine])
    
    pltSalesDetail.add_tools(hoverLine)
    
    pltSalesDetail.legend.click_policy = 'hide'
    pltSalesDetail.legend.title = 'Interactive Legend'
    
    minDate, maxDate = determineMinAndMaxDates(detailSalesPlotDataWins['Date'], detailSalesPlotDataLosses['Date'])
    
    detailedSales_DateSlider =  DateRangeSlider(start = minDate, end = maxDate,
                                         value = (minDate, maxDate),
                                         step = 1, title = 'Minimum and Maximum Dates Observed')
    
    callback_Slider_DS = CustomJS(args = dict(sourceDetailSumsWins = sourceDetailSumsWins, 
                                    sourceDetailSumsLosses = sourceDetailSumsLosses,
                                    sourceDetailSumsTotal = sourceDetailSumsTotal,
                                    sourceDetailSumsWins_Copy = sourceDetailSumsWins_Copy, 
                                    sourceDetailSumsLosses_Copy = sourceDetailSumsLosses_Copy,
                                    sourceDetailSumsTotal_Copy = sourceDetailSumsTotal_Copy,
                                    detailedSales_DateSlider = detailedSales_DateSlider
                                    ),
                        code = '''
                        var minDate = detailedSales_DateSlider.value[0];
                        var maxDate = detailedSales_DateSlider.value[1];
                        
                        const winsLineLength = sourceDetailSumsWins_Copy.data['Date'].length;
                        const lossesLineLength = sourceDetailSumsLosses_Copy.data['Date'].length;
                        const totalLineLength = sourceDetailSumsTotal_Copy.data['Date'].length;
                        
                        secondCopyDataSource_W = {'Extended Price Sum': [],
                                                  'Date': []
                                                  };
                        
                        secondCopyDataSource_L =  {'Extended Price Sum': [],
                                                  'Date': []
                                                  };
                        
                        secondCopyDataSource_T =  {'Extended Price Sum': [],
                                                  'Date': []
                                                  };
                        
                        for (i = 0; i < winsLineLength; i++){
                                if (sourceDetailSumsWins_Copy.data['Date'][i] >= minDate &&
                                sourceDetailSumsWins_Copy.data['Date'][i] <= maxDate){
                                        secondCopyDataSource_W['Extended Price Sum'].push(sourceDetailSumsWins_Copy.data['Extended Price Sum'][i]);
                                        secondCopyDataSource_W['Date'].push(sourceDetailSumsWins_Copy.data['Date'][i]);
                                }
                        }
                        
                        for (i = 0; i < lossesLineLength; i++){
                                if (sourceDetailSumsLosses_Copy.data['Date'][i] >= minDate &&
                                sourceDetailSumsLosses_Copy.data['Date'][i] <= maxDate){
                                        secondCopyDataSource_L['Extended Price Sum'].push(sourceDetailSumsLosses_Copy.data['Extended Price Sum'][i]);
                                        secondCopyDataSource_L['Date'].push(sourceDetailSumsLosses_Copy.data['Date'][i]);
                                }
                        }
                        
                        for (i = 0; i < totalLineLength; i++){
                                if (sourceDetailSumsTotal_Copy.data['Date'][i] >= minDate &&
                                sourceDetailSumsTotal_Copy.data['Date'][i] <= maxDate){
                                        secondCopyDataSource_T['Extended Price Sum'].push(sourceDetailSumsTotal_Copy.data['Extended Price Sum'][i]);
                                        secondCopyDataSource_T['Date'].push(sourceDetailSumsTotal_Copy.data['Date'][i]);
                                }
                        }
                        
                        console.log(secondCopyDataSource_W);
                        console.log(secondCopyDataSource_L);
                        console.log(secondCopyDataSource_T);
                        
                        sourceDetailSumsWins.data = {
                                ['Extended Price Sum']: secondCopyDataSource_W['Extended Price Sum'],
                                ['Date']: secondCopyDataSource_W['Date']
                                };
                        
                        sourceDetailSumsLosses.data = {
                                ['Extended Price Sum']: secondCopyDataSource_L['Extended Price Sum'],
                                ['Date']: secondCopyDataSource_L['Date']
                                };
                                
                        sourceDetailSumsTotal.data = {
                                ['Extended Price Sum']: secondCopyDataSource_T['Extended Price Sum'],
                                ['Date']: secondCopyDataSource_T['Date']
                                };
                        
                        sourceDetailSumsWins.change.emit();
                        sourceDetailSumsLosses.change.emit();
                        sourceDetailSumsTotal.change.emit();
                        ''') 
    
    detailedSales_DateSlider.js_on_change('value', callback_Slider_DS)
    
    return(pltSalesSum, pltSalesDetail, radio_Button_Group_MonthSum,  detailedSales_DateSlider)
    
    
def renderGraphs(plotDataWins, plotDataLosses, monthYearArray, alphabSortUniqueCompanyList, 
                 product_Groupings_Dict, item_Master_List, companyProductGroupItem_Breakdown):
    
    monthlyUniqueItemsTotal = {}
    for mY in monthYearArray:
        monthlyUniqueItemsTotal[mY] = []
        
    sumsPerMonthYearWins, uniqueItemsTotalPerMonth = \
        computingSumsPerMonthYear(plotDataWins, monthYearArray, monthlyUniqueItemsTotal)
    sumsPerMonthYearLosses, finishedUniqueItemsTotalPerMonth = \
        computingSumsPerMonthYear(plotDataLosses, monthYearArray, uniqueItemsTotalPerMonth)
    detailSalesPlotDataWins = quoteDataForDetailSales(plotDataWins)
    detailSalesPlotDataLosses = quoteDataForDetailSales(plotDataLosses)
    plt_Sales_Monthly, plt_Sales_Detail, radio_Button_Group_MonthlySales, \
        sliderDates_DetailedSales = salesLineGraph(detailSalesPlotDataWins, 
                                                 detailSalesPlotDataLosses, 
                                                 sumsPerMonthYearWins, 
                                                 sumsPerMonthYearLosses, 
                                                 monthYearArray, 
                                                 finishedUniqueItemsTotalPerMonth)
    
    meanStdDevPlotDataWins = computingMeansAndStdDeviation(plotDataWins, monthYearArray)
    meanStdDevPlotDataLosses = computingMeansAndStdDeviation(plotDataLosses, monthYearArray)
    plt_Means, radio_Button_Group_Means = marginMeansAndStdDeviationPlot(meanStdDevPlotDataWins, 
                                                                         meanStdDevPlotDataLosses,
                                                                         monthYearArray)
    
    tab2 = scatterPlotGraph(plotDataWins, plotDataLosses, alphabSortUniqueCompanyList, 
                            product_Groupings_Dict, item_Master_List, companyProductGroupItem_Breakdown)
    
    subTabSalesMonthly = Panel(child = column(plt_Sales_Monthly, widgetbox(radio_Button_Group_MonthlySales)), 
                               title = 'Sales Won And Lost Monthly')
    subTabSalesDetailed = Panel(child = column(plt_Sales_Detail, widgetbox(sliderDates_DetailedSales)),
                               title = 'Detailed Sales')
    subTabsSales = Tabs(tabs = [subTabSalesMonthly, subTabSalesDetailed])
    
    tab1 = Panel(child = column(plt_Means, widgetbox(radio_Button_Group_Means)), 
                 title = 'Means And Standard Deviation Plot')
    
    tab3 = Panel(child = subTabsSales,
                title = 'Sales Graphs')
    
    tabs = Tabs(tabs = [tab1, tab2, tab3])
    
    output_file(filename = 'Quote Activity Based Multi-Graph Composite.html', 
                title = 'Quote Activity Based Bokeh Composite Graphs',
                mode = 'inline')
    save(tabs)




#MAIN PROGRAM BELOW
#Above are functions used in script, below is main body of code
startTime = time.time()

items_File = 'itemMasterList.xlsx'
dataframeItems = pd.read_excel(items_File, sheet_name = 'ITEM CODE MASTER LIST')
item_Master_List = dataframeItems['Master List'].tolist()

product_Groupings_File = 'MOCKProductGroups.xlsx'
dataframeProductGroupings = pd.read_excel(product_Groupings_File, sheet_name = 'Product Group Lists')
product_Groups = list(dataframeProductGroupings.columns.values)

product_Groupings_Dict = {}
for group in product_Groups:
    product_Groupings_Dict[group] = dataframeProductGroupings[group].tolist()
    if isinstance(product_Groupings_Dict[group][-1], float):
        if math.isnan(product_Groupings_Dict[group][-1]):
            product_Groupings_Dict[group].pop(-1)

file = 'graphPlotDataWins.xlsx'
       
dataframeSheetPlotData = pd.read_excel(file, sheet_name = 'Wins Graph Plot Data')
    
companyUnderMonthListInWorkSheetWins = dataframeSheetPlotData['Company Under Month'].tolist()
itemCodeListInWorkSheetWins = dataframeSheetPlotData['Item Code'].tolist()
unitMarginListInWorkSheetWins = dataframeSheetPlotData['Unit Margin'].tolist()
quantityListInWorkSheetWins = dataframeSheetPlotData['Quantity'].tolist()
dateListInWorkSheetWins = dataframeSheetPlotData['Date'].tolist()
extendedPriceListInWorkSheetWins = dataframeSheetPlotData['Extended Price'].tolist()
    
dataframeSheetMonthYearDateList = pd.read_excel(file, sheet_name = 'Date List In Column')
    
monthYearList = dataframeSheetMonthYearDateList['Date'].tolist()
    
companyListInWorkSheetWins = companyUnderMonthListInWorkSheetWins
for monthYear in monthYearList:
    indexOfValueToRemove = companyListInWorkSheetWins.index(monthYear)
    companyListInWorkSheetWins.remove(monthYear)  
    itemCodeListInWorkSheetWins.pop(indexOfValueToRemove)
    unitMarginListInWorkSheetWins.pop(indexOfValueToRemove)
    quantityListInWorkSheetWins.pop(indexOfValueToRemove)
    dateListInWorkSheetWins.pop(indexOfValueToRemove)
    extendedPriceListInWorkSheetWins.pop(indexOfValueToRemove)
     
file = 'graphPlotDataLosses.xlsx'

dataframeSheetPlotData = pd.read_excel(file, sheet_name = 'Losses Graph Plot Data')
    
companyUnderMonthListInWorkSheetLosses = dataframeSheetPlotData['Company Under Month'].tolist()
itemCodeListInWorkSheetLosses = dataframeSheetPlotData['Item Code'].tolist()
unitMarginListInWorkSheetLosses = dataframeSheetPlotData['Unit Margin'].tolist()
quantityListInWorkSheetLosses = dataframeSheetPlotData['Quantity'].tolist()
dateListInWorkSheetLosses = dataframeSheetPlotData['Date'].tolist()
extendedPriceListInWorkSheetLosses = dataframeSheetPlotData['Extended Price'].tolist()
    
companyListInWorkSheetLosses = companyUnderMonthListInWorkSheetLosses
for monthYear in monthYearList:
    indexOfValueToRemove = companyListInWorkSheetLosses.index(monthYear)
    companyListInWorkSheetLosses.remove(monthYear)  
    itemCodeListInWorkSheetLosses.pop(indexOfValueToRemove)
    unitMarginListInWorkSheetLosses.pop(indexOfValueToRemove)
    quantityListInWorkSheetLosses.pop(indexOfValueToRemove)
    dateListInWorkSheetLosses.pop(indexOfValueToRemove)
    extendedPriceListInWorkSheetLosses.pop(indexOfValueToRemove)

#Common code for both wins and losses
allUniqueCompanyNames = list(set(companyListInWorkSheetWins + companyListInWorkSheetLosses))
alphabeticallySortedUniqueCompanyNamesList = sorted(allUniqueCompanyNames)
alphabeticallySortedUniqueCompanyNamesList.insert(0, 'ALL')

graphPlotDataWinsTotal = []
graphPlotDataLossesTotal = []

for index, company in enumerate(companyListInWorkSheetWins):
    graphPlotDataWinsTotal.append([company, itemCodeListInWorkSheetWins[index], 
                                   unitMarginListInWorkSheetWins[index],
                                   quantityListInWorkSheetWins[index], 
                                   dateListInWorkSheetWins[index], 
                                   extendedPriceListInWorkSheetWins[index]])
   
for index, company in enumerate(companyListInWorkSheetLosses):    
    graphPlotDataLossesTotal.append([company, itemCodeListInWorkSheetLosses[index], 
                                     unitMarginListInWorkSheetLosses[index],
                                     quantityListInWorkSheetLosses[index], 
                                     dateListInWorkSheetLosses[index],
                                     extendedPriceListInWorkSheetLosses[index]])


graphPlotDataWinsTotal.sort(key = lambda x: datetime.strptime(x[4], '%m-%d-%Y'))
graphPlotDataLossesTotal.sort(key = lambda x: datetime.strptime(x[4], '%m-%d-%Y'))

companyProductGroupItem_Breakdown = {}

dictionaryForDataFeedWinsTotal = {'Company': [], 'Item Code': [], 'Unit Margin': [], 'Quantity': [],
                                  'Date': [], 'Extended Price': [], 'Product Group': []}

appendCount = 0 #TESTING

for data in graphPlotDataWinsTotal:
    dictionaryForDataFeedWinsTotal['Company'].append(data[0])
    dictionaryForDataFeedWinsTotal['Item Code'].append(data[1])
    dictionaryForDataFeedWinsTotal['Unit Margin'].append(data[2])
    dictionaryForDataFeedWinsTotal['Quantity'].append(data[3])
    dictionaryForDataFeedWinsTotal['Date'].append(datetime.strptime(data[4] , '%m-%d-%Y'))
    dictionaryForDataFeedWinsTotal['Extended Price'].append(data[5])
    for key in product_Groups:
        if data[1] in product_Groupings_Dict[key]:
            keyAppended = key
            dictionaryForDataFeedWinsTotal['Product Group'].append(keyAppended)
            appendCount += 1 #TESTING
            
    if data[0] not in companyProductGroupItem_Breakdown:
        companyProductGroupItem_Breakdown[data[0]] = {keyAppended: [data[1]]}
    else:
        if keyAppended not in companyProductGroupItem_Breakdown[data[0]]:
            companyProductGroupItem_Breakdown[data[0]][keyAppended] = [data[1]]
        else:
            if data[1] not in companyProductGroupItem_Breakdown[data[0]][keyAppended]:
                companyProductGroupItem_Breakdown[data[0]][keyAppended].append(data[1])
        
print(appendCount, '    <--- TESTING\n') #TESTING
uniqueItems_W = set(dictionaryForDataFeedWinsTotal['Item Code']) #TESTING

        
dictionaryForDataFeedLossesTotal = {'Company': [], 'Item Code': [], 'Unit Margin': [], 'Quantity': [], 
                                    'Date': [], 'Extended Price': [], 'Product Group': []}

appendCount = 0 #TESTING

for data in graphPlotDataLossesTotal:
    dictionaryForDataFeedLossesTotal['Company'].append(data[0])
    dictionaryForDataFeedLossesTotal['Item Code'].append(data[1])
    dictionaryForDataFeedLossesTotal['Unit Margin'].append(data[2])
    dictionaryForDataFeedLossesTotal['Quantity'].append(data[3])
    dictionaryForDataFeedLossesTotal['Date'].append(datetime.strptime(data[4] , '%m-%d-%Y'))
    dictionaryForDataFeedLossesTotal['Extended Price'].append(data[5])
    for key in product_Groups:
        if data[1] in product_Groupings_Dict[key]:
            keyAppended = key
            dictionaryForDataFeedLossesTotal['Product Group'].append(keyAppended)
            appendCount += 1 #TESTING
            
    if data[0] not in companyProductGroupItem_Breakdown:
        companyProductGroupItem_Breakdown[data[0]] = {keyAppended: [data[1]]}
    else:
        if keyAppended not in companyProductGroupItem_Breakdown[data[0]]:
            companyProductGroupItem_Breakdown[data[0]][keyAppended] = [data[1]]
        else:
            if data[1] not in companyProductGroupItem_Breakdown[data[0]][keyAppended]:
                companyProductGroupItem_Breakdown[data[0]][keyAppended].append(data[1])

print(appendCount, '    <--- TESTING\n') #TESTING
uniqueItems_L = set(dictionaryForDataFeedLossesTotal['Item Code']) #TESTING

length_uniqueItems = len(set(list(uniqueItems_W) + list(uniqueItems_L)))
print(length_uniqueItems, '  UNIQUE ITEMS SET LENGTH  ::::: TESTING\n') #TESTING


renderGraphs(dictionaryForDataFeedWinsTotal, dictionaryForDataFeedLossesTotal, monthYearList, 
             alphabeticallySortedUniqueCompanyNamesList, product_Groupings_Dict, item_Master_List,
             companyProductGroupItem_Breakdown)

endTime = time.time()
programTime = endTime - startTime
print('****PROGRAM RUN TIME****:', programTime, '\n')