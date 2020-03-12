# coding: utf-8
import requests, time, json, os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

class ModelManager:

    def __init__(self, NNoutputNumber=None, optimizer='rmsprop', layers = 1, dropout = 0.0):
        self.outputNumber = NNoutputNumber
        self.optimizer = optimizer
        self.layers = layers
        self.dropout = dropout

    def setNNoutout(self, outputNumber):
        self.outputNumber = outputNumber

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setLayers(self, layers):
        self.layers = layers

    def setDropout(self, dropout):
        self.dropout = dropout

    def fetchTwSharePrice(self, stockNumber, targetMonth=17):

        nowYear = datetime.now().strftime('%Y')
        nowMonth = datetime.now().strftime('%m')

        month = int(nowMonth)
        year = int(nowYear)
        tableDict = {}
        for t in range(targetMonth):

            if month == 0:
                month = 12
                year -= 1

            if month < 10:
                month = '0' + str(month)

            url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={year}{month}01&stockNo={stockNumber}".format(
                stockNumber=stockNumber, year=year, month=month)
            print(url)
            time.sleep(3)

            jsonData = json.loads(requests.get(url).content)
            i = 0
            try:
                stockInfoList = jsonData['data']
                print(stockInfoList)
            except:
                i += 1
                month = int(month)
                month -= i
                if month < 10:
                    month = '0' + str(month)
                url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={year}{month}01&stockNo={stockNumber}".format(
                    stockNumber=stockNumber, year=year, month=month)
                jsonData = json.loads(requests.get(url).content)
                stockInfoList = jsonData['data']
                print(stockInfoList)

            dates = []
            tradedQtys = []
            tradedPrices =[]
            opens = []
            peaks =[]
            bottoms =[]
            prices = []
            diffs =[]
            vols = []

            stockInfoList = jsonData['data']
            for stockInfo in stockInfoList:
                rocYear, rocMonth, rocDate = stockInfo[0].split('/')
                # date = str(int(rocYear) + 1911) + rocMonth + rocDate
                date = "{year}-{month}-{date}".format(year=str(int(rocYear) + 1911), month=rocMonth, date=rocDate)
                dates.append(date)
                tradedQty = stockInfo[1]
                tradedQtys.append(tradedQty)
                tradedPrice = stockInfo[2]
                tradedPrices.append(tradedPrice)
                open = stockInfo[3]
                opens.append(open)
                peak = stockInfo[4]
                peaks.append(peak)
                bottom = stockInfo[5]
                bottoms.append(bottom)
                price = stockInfo[6]
                prices.append(price)
                diff = stockInfo[7]
                diffs.append(diff)
                vol = stockInfo[8]
                vols.append(vol)


            dates.reverse()
            prices.reverse()
            summery = {
                "date": dates,
                "tradedQty": tradedQtys,
                "tradedPrice": tradedPrices,
                "open": opens,
                "high": peaks,
                "low": bottoms,
                "close": prices,
                "diff": diffs,
                "vol": vols
            }

            table = pd.DataFrame(summery)
            tableDict['{year}/{month}'.format(year=year, month=month)] = table
            print(tableDict['{year}/{month}'.format(year=year, month=month)])

            month = int(month)
            month -= 1

        allTable = []

        for tableKey in tableDict:
            allTable.append(tableDict[tableKey])

        combinedTable = pd.concat(allTable)

        combinedTable.index = range(len(combinedTable.date))
        directory = 'rawData'
        file = 'twSharePrice-{stockNumber}-{date}.xlsx'.format(date=datetime.now().strftime("%Y%m%d"), stockNumber=stockNumber)
        fileCsv = 'twSharePrice-{date}.csv'.format(date=datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(directory):
            os.mkdir(directory)

        combinedTable.to_excel(os.path.join(directory, file))
        combinedTable.to_csv(os.path.join(directory, fileCsv))

        return combinedTable


    def fetchDataForNN(self, stockNumber):
        try:
            if sys.platform == "linux":
                driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")  # Linux
            elif sys.platform == "darwin":
                driver = webdriver.Chrome("/usr/local/bin/chromedriver")  # for MACOS
            else:
                driver = webdriver.Chrome("C:\\Users\\Eric\\Documents\\Python\\chromedriver")  # for Windows

            wait = WebDriverWait(driver, 10)

            url = "https://www.tdcc.com.tw/smWeb/QryStock.jsp"

            priceTable = self.fetchTwSharePrice(stockNumber=stockNumber)

            # Feed url to Chrome driver and sleep 3 secs
            driver.get(url)

            time.sleep(3)

            # use BeautifulSoup to parse html (detail see BeautifulSoup documentation) to find out avaiable date
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # create relevant Varibles needed:
            between = []
            preGroup1 = None
            diffGroup1 = []
            preGroup2 = None
            diffGroup2 = []
            preGroup3 = None
            diffGroup3 = []
            preGroup4 = None
            diffGroup4 = []
            preGroup5 = None
            diffGroup5 = []
            preGroup6 = None
            diffGroup6 = []
            preGroup7 = None
            diffGroup7 = []
            preGroup8 = None
            diffGroup8 = []
            preGroup9 = None
            diffGroup9 = []
            preGroup10 = None
            diffGroup10 = []
            preGroup11 = None
            diffGroup11 = []
            preGroup12 = None
            diffGroup12 = []
            preGroup13 = None
            diffGroup13 = []
            preGroup14 = None
            diffGroup14 = []
            preGroup15 = None
            diffGroup15 = []
            thisWeekAvgPrice = []
            priceDiffNextWeek = []
            priceDiffPreOneWeek = []
            priceDiffPreTwoWeek = []
            priceDiffPreThreeWeek = []
            priceDiffPreMonth = []

            # parse html and get all available date which on the dropdown list

            weeks = soup.findAll("option")
            weekCount = 0
            for week in weeks:
                between.append("{week}".format(week=week.string))
                print(between)
                weekCount += 1

                # Calc the weekly average of stock price
                price5AfterBetween = []
                price5BeforeBetween = []
                price5to10BeforeBetween = []
                price10to15BeforeBetween = []
                price15to20BeforeBetween = []
                price20to25BeforeBetween = []
                weekday = between[-1]
                while priceTable[priceTable['date'] == weekday].size == 0:
                    year = weekday[:4]
                    month = int(weekday[4:-2])
                    date = int(weekday[-2:])
                    date = date - 1
                    if date == 0:
                        date = 31
                        month = month - 1
                    if date < 10:
                        date = "0" + str(date)
                    if month < 10:
                        month = "0" + str(month)
                    weekday = year + str(month) + str(date)

                if priceTable[priceTable['date'] == weekday].index > 5:
                    count = 0
                    while len(price5AfterBetween) < 5:
                        day0 = priceTable[priceTable['date'] == weekday].index[0]
                        price5AfterBetween.append(float(priceTable.price.iloc[day0 - count]))
                        price5BeforeBetween.append(float(priceTable.price.iloc[day0 + count]))
                        price5to10BeforeBetween.append(float(priceTable.price.iloc[day0 + 5 + count]))
                        price10to15BeforeBetween.append(float(priceTable.price.iloc[day0 + 10 + count]))
                        price15to20BeforeBetween.append(float(priceTable.price.iloc[day0 + 15 + count]))
                        price20to25BeforeBetween.append(float(priceTable.price.iloc[day0 + 20 + count]))
                        count += 1
                    avgPriceNext5 = sum(price5AfterBetween) / 5
                else:
                    count = 0
                    while len(price5BeforeBetween) < 5:
                        day0 = priceTable[priceTable['date'] == weekday].index[0]
                        price5BeforeBetween.append(float(priceTable.price.iloc[day0 + count]))
                        price5to10BeforeBetween.append(float(priceTable.price.iloc[day0 + 5 + count]))
                        price10to15BeforeBetween.append(float(priceTable.price.iloc[day0 + 10 + count]))
                        price15to20BeforeBetween.append(float(priceTable.price.iloc[day0 + 15 + count]))
                        price20to25BeforeBetween.append(float(priceTable.price.iloc[day0 + 20 + count]))
                        count += 1

                avgPricePre5 = sum(price5BeforeBetween) / 5
                thisWeekAvgPrice.append(avgPricePre5)
                avgPrice5to10 = sum(price5to10BeforeBetween) / 5
                avgPrice10to15 = sum(price10to15BeforeBetween) / 5
                avgPrice15to20 = sum(price15to20BeforeBetween) / 5
                avgPrice20to25 = sum(price20to25BeforeBetween) / 5

                try:
                    priceDiffNext5 = round((avgPriceNext5 - avgPricePre5) / avgPricePre5, 5)
                except:
                    priceDiffNext5 = 0
                priceDiff5to10 = round((avgPricePre5 - avgPrice5to10) / avgPrice5to10, 5)
                priceDiff10to15 = round((avgPricePre5 - avgPrice10to15) / avgPrice10to15, 5)
                priceDiff15to20 = round((avgPricePre5 - avgPrice15to20) / avgPrice15to20, 5)
                priceDiff20to25 = round((avgPricePre5 - avgPrice20to25) / avgPrice20to25, 5)

                priceDiffNextWeek.append(priceDiffNext5)
                priceDiffPreOneWeek.append(priceDiff5to10)
                priceDiffPreTwoWeek.append(priceDiff10to15)
                priceDiffPreThreeWeek.append(priceDiff15to20)
                priceDiffPreMonth.append(priceDiff20to25)
                print(priceDiffNextWeek)
                print(priceDiffPreOneWeek)
                print(priceDiffPreTwoWeek)
                print(priceDiffPreThreeWeek)
                print(priceDiffPreMonth)

                # Manipulate javascript to:  fill in needed date --> stock number --> click search button
                driver.execute_script('document.getElementsByName("scaDate")[0].value="{week}"'.format(week=week.string))
                time.sleep(1)
                driver.execute_script('document.getElementById("StockNo").value = "{stock}"'.format(stock=stockNumber))
                time.sleep(1)
                driver.execute_script('document.getElementsByName("sub")[0].click()')
                time.sleep(2)

                # use BeautifulSoup to parse through html (detail see BeautifulSoup documentation) to fetch shareholder Info
                soup = BeautifulSoup(driver.page_source, "html.parser")

                stockInfos = soup.findAll("tbody")[7].findAll("tr")
                print(len(stockInfos))
                # retry untill it gets correct html
                while len(stockInfos) < 17:
                    driver.execute_script(
                        'document.getElementsByName("scaDate")[0].value="{week}"'.format(week=week.string))
                    time.sleep(1)
                    driver.execute_script('document.getElementById("StockNo").value = "{stock}"'.format(stock=stockNumber))
                    time.sleep(1)
                    driver.execute_script('document.getElementsByName("sub")[0].click()')
                    time.sleep(2)

                    soup = BeautifulSoup(driver.page_source, "html.parser")

                    stockInfos = soup.findAll("tbody")[7].findAll("tr")

                count = 1
                print(len(stockInfos))

                # fetch the shareholder data and calculate the diff data
                for stockInfo in stockInfos:
                    try:
                        data = float(stockInfo.findAll("td")[4].string)
                        if count == 1:
                            if preGroup1 is None:
                                preGroup1 = data
                            else:
                                diffGroup1.append(round(preGroup1 - data, 3))
                                preGroup1 = data
                            print(diffGroup1)

                        if count == 2:
                            if preGroup2 is None:
                                preGroup2 = data
                            else:
                                diffGroup2.append(round(preGroup2 - data, 3))
                                preGroup2 = data
                            print(diffGroup2)
                        if count == 3:
                            if preGroup3 is None:
                                preGroup3 = data
                            else:
                                diffGroup3.append(round(preGroup3 - data, 3))
                                preGroup3 = data
                            print(diffGroup3)

                        if count == 4:
                            if preGroup4 is None:
                                preGroup4 = data
                            else:
                                diffGroup4.append(round(preGroup4 - data, 3))
                                preGroup4 = data
                            print(diffGroup4)

                        if count == 5:
                            if preGroup5 is None:
                                preGroup5 = data
                            else:
                                diffGroup5.append(round(preGroup5 - data, 3))
                                preGroup5 = data
                            print(diffGroup5)

                        if count == 6:
                            if preGroup6 is None:
                                preGroup6 = data
                            else:
                                diffGroup6.append(round(preGroup6 - data, 3))
                                preGroup6 = data
                            print(diffGroup6)

                        if count == 7:
                            if preGroup7 is None:
                                preGroup7 = data
                            else:
                                diffGroup7.append(round(preGroup7 - data, 3))
                                preGroup7 = data
                            print(diffGroup7)

                        if count == 8:
                            if preGroup8 is None:
                                preGroup8 = data
                            else:
                                diffGroup8.append(round(preGroup8 - data, 3))
                                preGroup8 = data
                            print(diffGroup8)
                        if count == 9:
                            if preGroup9 is None:
                                preGroup9 = data
                            else:
                                diffGroup9.append(round(preGroup9 - data, 3))
                                preGroup9 = data
                            print(diffGroup9)
                        if count == 10:
                            if preGroup10 is None:
                                preGroup10 = data
                            else:
                                diffGroup10.append(round(preGroup10 - data, 3))
                                preGroup10 = data
                            print(diffGroup10)
                        if count == 11:
                            if preGroup11 is None:
                                preGroup11 = data
                            else:
                                diffGroup11.append(round(preGroup11 - data, 3))
                                preGroup11 = data
                            print(diffGroup11)
                        if count == 12:
                            if preGroup12 is None:
                                preGroup12 = data
                            else:
                                diffGroup12.append(round(preGroup12 - data, 3))
                                preGroup12 = data
                            print(diffGroup12)
                        if count == 13:
                            if preGroup13 is None:
                                preGroup13 = data
                            else:
                                diffGroup13.append(round(preGroup13 - data, 3))
                                preGroup13 = data
                            print(diffGroup13)
                        if count == 14:
                            if preGroup14 is None:
                                preGroup14 = data
                            else:
                                diffGroup14.append(round(preGroup14 - data, 3))
                                preGroup14 = data
                            print(diffGroup14)
                        if count == 15:
                            if preGroup15 is None:
                                preGroup15 = data
                            else:
                                diffGroup15.append(round(preGroup15 - data, 3))
                                preGroup15 = data
                            print(diffGroup15)
                            print(len(diffGroup15))

                        count += 1

                    except:
                        print("Table Title bypass")

                print(len(between), len(diffGroup1),len(diffGroup2),len(diffGroup3),len(diffGroup4),len(diffGroup5),len(diffGroup6),len(diffGroup7),len(diffGroup8),len(diffGroup9),len(diffGroup10),len(diffGroup11),len(diffGroup12),len(diffGroup13),len(diffGroup14),len(diffGroup15))
                print("========================================================================")

            diffGroup1.append(None)
            diffGroup2.append(None)
            diffGroup3.append(None)
            diffGroup4.append(None)
            diffGroup5.append(None)
            diffGroup6.append(None)
            diffGroup7.append(None)
            diffGroup8.append(None)
            diffGroup9.append(None)
            diffGroup10.append(None)
            diffGroup11.append(None)
            diffGroup12.append(None)
            diffGroup13.append(None)
            diffGroup14.append(None)
            diffGroup15.append(None)
            print("len of diffGroup1: {len}".format(len=len(diffGroup1)))
            print("len of diffGroup2: {len}".format(len=len(diffGroup2)))
            print("len of diffGroup3: {len}".format(len=len(diffGroup3)))
            print("len of diffGroup4: {len}".format(len=len(diffGroup4)))
            print("len of diffGroup5: {len}".format(len=len(diffGroup5)))
            print("len of diffGroup6: {len}".format(len=len(diffGroup6)))
            print("len of diffGroup7: {len}".format(len=len(diffGroup7)))
            print("len of diffGroup8: {len}".format(len=len(diffGroup8)))
            print("len of diffGroup9: {len}".format(len=len(diffGroup9)))
            print("len of diffGroup10: {len}".format(len=len(diffGroup10)))
            print("len of diffGroup11: {len}".format(len=len(diffGroup11)))
            print("len of diffGroup12: {len}".format(len=len(diffGroup12)))
            print("len of diffGroup13: {len}".format(len=len(diffGroup13)))
            print("len of diffGroup14: {len}".format(len=len(diffGroup14)))
            print("len of diffGroup15: {len}".format(len=len(diffGroup15)))
            print("len of between: {len}".format(len=len(between)))

            """
            ---------------Create panda format Dataframe and export to excel ----------------------
            """

            summery = {
                "week": between,
                "thisWeekAvgPrice": thisWeekAvgPrice,
                "priceDiffNextWeek": priceDiffNextWeek,
                "priceDiff1WeekBefore": priceDiffPreOneWeek,
                "priceDiff2WeeksBefore": priceDiffPreTwoWeek,
                "priceDiff3WeeksBefore": priceDiffPreThreeWeek,
                "priceDiff1MonthBefore": priceDiffPreMonth,
                "1,000-5,000": diffGroup2,
                "5,001-10,000": diffGroup3,
                "10,001-15,000": diffGroup4,
                "15,001-20,000": diffGroup5,
                "20,001-30,000": diffGroup6,
                "30,001-40,000": diffGroup7,
                "40,001-50,000": diffGroup8,
                "50,001-100,000": diffGroup9,
                "100,001-200,000": diffGroup10,
                "200,001-400,000": diffGroup11,
                "400,001-600,000": diffGroup12,
                "600,001-800,000": diffGroup13,
                "800,001-1,000,000": diffGroup14,
                "1,000,001以上": diffGroup15
            }

            summeryTable = pd.DataFrame(summery).dropna()
            print(summeryTable)
            print("got the summery Table raw")
            rawDataDir = 'rawData'
            if not os.path.exists(rawDataDir):
                os.mkdir(rawDataDir)

            date = datetime.strptime(between[0],'%Y%m%d')
            print(date)
            try:
                lastweek = date - timedelta(7)
                print(lastweek)
                fileNameCsvLastWeek = "StockANN_raw_{stockNumber}_{date}_v1.csv".format(stockNumber=stockNumber,
                                                                                date=lastweek.strftime('%Y-wk%W'))
                summeryTableLastWeek = pd.read_csv(os.path.join(rawDataDir,fileNameCsvLastWeek)).dropna().astype(str)
                print(summeryTableLastWeek)
                print("following is droping unnamed field")
                try:
                    print("tring to drop unnamed field")
                    summeryTableLastWeek = summeryTableLastWeek.drop(['Unnamed: 0'], axis=1)
                    print("unneeded field droped")
                    print(summeryTableLastWeek)
                except:
                    print("no unnamed field to drop")
                print(summeryTableLastWeek)
                dfDates = np.array(summeryTable['week'])
                #print(dfDates)
                df2Dates = np.array(summeryTableLastWeek['week'])
                #print(df2Dates)
                diffDates = []
                for df2Date in df2Dates:
                    if df2Date not in dfDates:
                        diffDates.append(df2Date)
                        print(diffDates)
                conTable = []
                for diffDate in diffDates:
                    mask = summeryTableLastWeek['week'] == diffDate
                    conTable.append(summeryTableLastWeek[mask])
                    print(summeryTableLastWeek[mask])
                diffline = pd.concat(conTable)
                print(diffline)
                summeryTable = pd.concat([summeryTable, diffline])

                print(summeryTable)
                summeryTableIndex = np.array(range(summeryTable.shape[0]))
                summeryTable.index = summeryTableIndex
                print(summeryTable)
            except:
                print("nothing to combine")

            print("next step")
            fileNameExcel = "StockANN_raw_{stockNumber}_{date}_v1.xlsx".format(stockNumber=stockNumber, date=date.strftime('%Y-wk%W'))
            fileNameCsv = "StockANN_raw_{stockNumber}_{date}_v1.csv".format(stockNumber=stockNumber, date=date.strftime('%Y-wk%W'))
            print(fileNameCsv)
            summeryTable.to_excel(os.path.join(rawDataDir, fileNameExcel), sheet_name="sheet1")
            summeryTable.to_csv(os.path.join(rawDataDir, fileNameCsv))
            driver.quit()
            print(summeryTable)
            return summeryTable
        except:
            driver.quit()
            print("fail to fetch data: internet?")

    def build_classifier(self, inputNumber=18):
        outputNumber = self.outputNumber
        optimizer = self.optimizer
        input_dim = int(inputNumber) #x_train.shape[1]
        output_dim_hidden = int(round((inputNumber + outputNumber) / 2, 0)) #y_train.shape[1]
        output_dim_out = int(outputNumber) #y_train.shape[1]
        if output_dim_out == 1:
            activation_out = 'sigmoid'
            compile_loss = 'binary_crossentropy'
        else:
            activation_out = 'softmax'
            compile_loss = 'categorical_crossentropy'

        classifier = Sequential()

        classifier.add(Dense(output_dim=output_dim_hidden, init='uniform', activation='relu', input_dim=input_dim))
        classifier.add(Dropout(0.0))

        for layer in range(self.layers):
            classifier.add(Dense(output_dim=output_dim_hidden, init='uniform', activation='relu'))
            classifier.add(Dropout(self.dropout))

        classifier.add(Dense(output_dim=output_dim_out, init='uniform', activation=activation_out))
        classifier.add(Dropout(0.0))

        classifier.compile(optimizer=optimizer, loss=compile_loss, metrics=['accuracy'])

        return classifier

    def gridSearch(self, batchs, nbs, optimizers, X, y, stockNumber, layers, dropouts, fileDate= datetime.now().strftime('%Y-wk%W')):

        kFoldMeans = []
        kFoldStds = []
        nbCount = []
        batchCount = []
        optimizerCount = []
        layerCount = []
        dropoutCount = []
        for layer in range(layers):
            self.setLayers(layer)
            for dropout in dropouts:
                self.setDropout(dropout)
                for optimizer in optimizers:
                    self.setOptimizer(optimizer)
                    for batch in batchs:
                        for nb in nbs:
                            classifier = KerasClassifier(build_fn=self.build_classifier, batch_size=batch, nb_epoch=nb)
                            accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=10, n_jobs=-1)
                            mean = accuracies.mean()
                            kFoldMeans.append(mean)
                            print(mean)
                            variance = accuracies.std()
                            kFoldStds.append(variance)
                            print(variance)
                            print(self.outputNumber)
                            nbCount.append(nb)
                            layerCount.append(layer+1)
                            dropoutCount.append(dropout)
                            batchCount.append(batch)
                            optimizerCount.append(optimizer)

        summery = {"layer": layerCount,
                   "dropout": dropoutCount,
                   "nb_epoch": nbCount,
                   "batch_size": batchCount,
                   "optimizer": optimizerCount,
                   "kFoldMeans": kFoldMeans,
                   "kFoldStds": kFoldStds}
        summeryTable = pd.DataFrame(summery)
        bestSelector = summeryTable['kFoldMeans'] == summeryTable['kFoldMeans'].max()
        bestSelectorIndex = summeryTable[bestSelector].index[0]
        bestLayer = summeryTable.iloc[bestSelectorIndex,0]
        bestDropout = summeryTable.iloc[bestSelectorIndex, 1]
        bestEpoch = summeryTable.iloc[bestSelectorIndex,2]
        print(bestEpoch)
        bestBatch = summeryTable.iloc[bestSelectorIndex, 3]
        print(bestBatch)
        bestOptimizer = summeryTable.iloc[bestSelectorIndex,4]
        print(bestOptimizer)
        bestMean = summeryTable.iloc[bestSelectorIndex,5]
        bestStd = summeryTable.iloc[bestSelectorIndex,6]

        directory = 'modelSelection'
        file = 'gridSearch_{stockNumber}_{date}.xlsx'.format(stockNumber=stockNumber, date= fileDate)
        if not os.path.exists(directory):
            os.mkdir(directory)

        summeryTable.to_excel(os.path.join(directory, file))
        return bestEpoch, bestBatch, bestOptimizer, bestMean, bestStd, bestLayer, bestDropout

    def predictSTockPrice(self):
        stocksSheet = pd.read_csv('stocksOnDemand.csv')
        print(stocksSheet)

        if stocksSheet['TestField'][0] == "TRUE":
            targetRange = [len(stocksSheet.columns)-1]
            print(targetRange)
        else:
            targetRange = list(range(len(stocksSheet.columns)-1))
            print(targetRange)

        for col in targetRange:

            stockNumberList = list(stocksSheet.iloc[:, col].dropna().astype(str))
            print(stockNumberList)
            stockNumberList.pop(0)
            resultLossList = []
            resultLess5List = []
            resultMore5List = []
            kFoldMeans = []
            kFoldVars = []
            fileDate = ""

            for stockNumber in stockNumberList:
                try:
                    importSourceDir = 'rawData'
                    importSourceFile = "StockANN_raw_{stockNumber}_{date}_v1.csv".format(
                        stockNumber=stockNumber, date=datetime.now().strftime('%Y-wk%W'))

                    if os.path.exists(os.path.join(importSourceDir,importSourceFile)):
                        df = pd.read_csv(os.path.join(importSourceDir, importSourceFile)).dropna()
                        fileDate = datetime.now().strftime('%Y-wk%W')
                        try:
                            df = df.drop(['Unnamed: 0'], axis=1)
                            print("unneeded field droped")
                        except:
                            print("combine old table without drop")
                    else:
                        updatedWeek = self.checkIfWebsiteUpdated()
                        if updatedWeek == datetime.now().strftime('%Y-wk%W'):
                            df = self.fetchDataForNN(stockNumber).dropna()
                            print(df)
                            fileDate = df['week'][0]
                            fileDate = datetime.strptime(str(fileDate), '%Y%m%d')
                            fileDate = fileDate.strftime('%Y-wk%W')
                            print(fileDate)

                        else:
                            wk = int(datetime.now().strftime('%W'))-1
                            if wk < 10:
                                wk = "0"+str(wk)
                                yr = datetime.now().strftime('%Y')
                                importSourceFile = "StockANN_raw_{stockNumber}_{yr}-wk{wk}_v1.csv".format(
                                    stockNumber=stockNumber, yr=yr,wk=wk)
                                df = pd.read_csv(os.path.join(importSourceDir, importSourceFile)).dropna()
                                try:
                                    df = df.drop(['Unnamed: 0'], axis=1)
                                    print("unneeded field droped")
                                except:
                                    print("combine old table without drop")
                                fileDate = "{yr}-wk{wk}".format(
                                    stockNumber=stockNumber, yr=yr,wk=wk)
                            else:
                                wk =str(wk)
                                yr = datetime.now().strftime('%Y')
                                importSourceFile = "StockANN_raw_{stockNumber}_{yr}-wk{wk}_v1.csv".format(
                                    stockNumber=stockNumber, yr=yr, wk=wk)
                                df = pd.read_csv(os.path.join(importSourceDir, importSourceFile)).dropna()
                                try:
                                    df = df.drop(['Unnamed: 0'], axis=1)
                                    print("unneeded field droped")
                                except:
                                    print("combine old table without drop")
                                fileDate = "{yr}-wk{wk}".format(
                                    stockNumber=stockNumber, yr=yr, wk=wk)


                    rawData = df.iloc[2:, :]
                    rawData = rawData.astype(float)
                    print(rawData.iloc[:, 2:].corr())

                    maskLessZero = rawData['priceDiffNextWeek'] < 0
                    rawData['priceDiffNextWeek'][maskLessZero] = 0

                    mask5persent = rawData['priceDiffNextWeek'] > 0.05

                    rawData['priceDiffNextWeek'][mask5persent] = 2

                    mask1 = rawData['priceDiffNextWeek'] < 0.05
                    mask2 = rawData['priceDiffNextWeek'] > 0

                    rawData['priceDiffNextWeek'][mask1 & mask2] = 1

                    rawData['priceDiffNextWeek'].astype(int)

                    priceDiffNextWeek = pd.get_dummies(rawData['priceDiffNextWeek'], prefix='priceDiffNextWeek')

                    xStartValue = 0
                    for i in range(len(rawData.columns)):
                        if rawData.columns[i] == "priceDiff1WeekBefore":
                            xStartValue = i

                    x = rawData.iloc[:, xStartValue:]
                    y = priceDiffNextWeek
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
                    sc = StandardScaler()
                    x_train = sc.fit_transform(x_train)
                    x_test = sc.fit_transform(x_test)

                    self.setNNoutout(y_train.shape[1])
                    bestEpoch, bestBatch, bestOptimizer, bestMean, bestStd, bestLayer, bestDropout = self.gridSearch(
                        [10, 15, 20], [200, 350],
                        ['rmsprop', 'adam'],
                        X=x_train, y=y_train,
                        stockNumber=stockNumber,
                        layers=2, dropouts=[0.0], fileDate = fileDate)
                    self.setOptimizer(bestOptimizer)
                    self.setLayers(bestLayer)
                    self.setDropout(bestDropout)
                    classifier = self.build_classifier(inputNumber=x_train.shape[1])

                    kFoldMean = round(bestMean, 3)
                    kFoldVar = round(bestStd, 3)

                    classifier.fit(x_train, y_train, batch_size=bestBatch, nb_epoch=bestEpoch)

                    y_pred = classifier.predict(x_test)

                    print(y_pred)

                    y_p = (y_pred > 0.5)

                    print(y_p)

                    print(y_test)

                    dataForNextWeek = df.iloc[0, xStartValue:]

                    dataForPredit = np.array(dataForNextWeek).reshape(1, -1)

                    nextWeek_pred = sc.fit_transform(dataForPredit)

                    result = classifier.predict(nextWeek_pred)
                    resultLoss = str(round(result[0][0], 2))
                    resultLess5 = str(round(result[0][1], 2))
                    if result.shape[1] == 3:
                        resultMore5 = str(round(result[0][2], 2))
                    else:
                        resultMore5 = 0

                    print(result)

                    print(result > 0.5)

                    resultLossList.append(resultLoss)
                    resultLess5List.append(resultLess5)
                    resultMore5List.append(resultMore5)
                    kFoldMeans.append(kFoldMean)
                    kFoldVars.append(kFoldVar)

                except:
                    print("fail in {stockNumber}".format(stockNumber=stockNumber))
                    resultLossList.append(None)
                    resultLess5List.append(None)
                    resultMore5List.append(None)
                    kFoldMeans.append(None)
                    kFoldVars.append(None)

            NNSummery = {
                "StockNumber": stockNumberList,
                "PriceDownNextWeek": resultLossList,
                "PriceUp0-5%NextWeek": resultLess5List,
                "priceUp>5%NextWeek": resultMore5List,
                "modelAccuracy": kFoldMeans,
                "modelpredictStd": kFoldVars
            }

            NNSummeryTable = pd.DataFrame(NNSummery)


            resultDataDir = 'resultData'

            if not os.path.exists(resultDataDir):
                os.mkdir(resultDataDir)
            fileName = "shareStockANN_result_{stockGroup}_{date}_v1.xlsx".format(
                stockGroup=stocksSheet.columns[col], date=fileDate)
            NNSummeryTable.to_excel(os.path.join(resultDataDir, fileName), sheet_name="sheet1")


    def checkIfWebsiteUpdated(self):
        if sys.platform == "linux":
            driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")  # Linux
        elif sys.platform == "darwin":
            driver = webdriver.Chrome("/usr/local/bin/chromedriver")  # for MACOS
        else:
            driver = webdriver.Chrome(
                "C:\\Users\\Eric\\Documents\\Python\\chromedriver")  # for Windows

        wait = WebDriverWait(driver, 3)

        url = "https://www.tdcc.com.tw/smWeb/QryStock.jsp"

        # Feed url to Chrome driver and sleep 3 secs
        driver.get(url)

        time.sleep(3)
        print("url load")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        updatedWeek = soup.findAll("option")[0].string
        print(updatedWeek)
        updatedWeek = datetime.strptime(updatedWeek, '%Y%m%d')
        updatedWeek = updatedWeek.strftime('%Y-wk%W')
        print(updatedWeek)
        print(datetime.now().strftime('%Y-wk%W'))
        return updatedWeek


    def fetchCurrency(self, id='USDTWD'):
        r = requests.get('https://tw.rter.info/capi.php')
        currency = r.json()

        currTime = datetime.now().strftime('%Y/%m/%d %H:%M')
        currencyLst = []
        for each in currency:
            if each[0:3] == 'USD':
                currencyLst.append(each)
            else:
                pass

        currencyCodeDict = dict()

        for line in currencyLst:
            currencyCodeDict[line] = round(currency[line]['Exrate'], 3)

        for each in currencyLst:
            if id[0:3] == each[3:]:
                USDfst3 = currencyCodeDict['USD' + str(id[0:3])]
                for each in currencyLst:
                    if id[3:] == each[3:]:
                        USDlst3 = currencyCodeDict['USD' + str(id[3:])]
                        currencyCode = [
                            round((USDlst3 / USDfst3), 4),
                            'Japen Time ' + str(currTime) + ' ' + str(id[0:3]) + ' vs ' + str(id[3:])
                        ]

        return currencyCode