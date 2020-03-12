
import time
from bs4 import BeautifulSoup
import sys, os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd

class AppReviewManager:

    def getAppleReview(self, urls, browserCheckoutCount=None):

        directory = "appReview"
        if not os.path.exists(directory):
            os.mkdir(directory)

        if sys.platform == "linux":
            driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")  # Linux
        elif sys.platform == "darwin":
            driver = webdriver.Chrome("/usr/local/bin/chromedriver")  # for MACOS
        else:
            driver = webdriver.Chrome("C:\\Users\\Eric\\Documents\\Python\\chromedriver")  # for Windows

        WebDriverWait(driver, 5)

        summeryTableList = []

        for url in urls:

            authors = []
            reviewDates = []
            ratings = []
            bodies = []

            driver.get(url)

            page = driver.page_source

            soup_expatistan = BeautifulSoup(page, "html.parser")

            expatistan_table = soup_expatistan.find("h1", class_="product-header__title app-header__title")
            app = expatistan_table.text
            appNameStr = str(app).replace("\n","-")
            appName = appNameStr.replace(" ", "")

            print("App name: " + app)


            expatistan_table = soup_expatistan.find("span", class_="we-customer-ratings__averages__display")
            # print("Rating Value: ", expatistan_table.text)

            expatistan_table = soup_expatistan.find("div", class_="we-customer-ratings__count small-hide medium-show")

            ratingCountRaw = expatistan_table.text
            ratingCount = ""

            for i in ratingCountRaw:
                try:
                    ratingCount += str(int(i))
                except:
                    if i == '.':
                        ratingCount += i

            if "万" in ratingCountRaw:
                # ratingCount = str(float(ratingCount)*10000)
                ratingCount = "10000"

            print(ratingCountRaw)
            print("Total Rating Count: ", ratingCount)
            existingFile = 'AppleReviewTable-{appName}.csv'.format(appName=appName)

            if os.path.exists(os.path.join(os.path.join(directory, existingFile))):
                reviewTable = pd.read_csv(os.path.join(os.path.join(directory, existingFile)))

            else:

                expatistan_table = soup_expatistan.find("div", class_="we-customer-ratings__count small-hide medium-show")

                # print("Reviews Count: ", expatistan_table.text)

                soup_histogram = soup_expatistan.find("div", class_="VEF2C")

                url = url + '#see-all/reviews'
                driver.get(url)
                time.sleep(5)  # wait dom ready
                bufferBody=""
                browserCheckCount = 0
                print("gogo")

                for i in range(1, round(int(float(ratingCount)/10))):
                    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')  # scroll to load other reviews
                    page = driver.page_source
                    soup_expatistan = BeautifulSoup(page, "html.parser")

                    if bufferBody == str(soup_expatistan.findAll("div",
                                                       class_='ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height')[-1]
                                                 .find("div", class_="we-clamp ember-view").p.string):
                        browserCheckCount += 1
                        print("scrollDownTime: {browserCheck}".format(browserCheck=browserCheckCount))
                    else:
                        bufferBody = str(soup_expatistan.findAll("div",
                                                       class_='ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height')[-1]
                                                 .find("div", class_="we-clamp ember-view").p.string)
                    try:
                        if browserCheckCount > browserCheckoutCount:
                            break
                    except:
                        pass

                    time.sleep(1.5)


                page = driver.page_source
                soup_expatistan = BeautifulSoup(page, "html.parser")
                expand_pages = soup_expatistan.findAll("div",
                                                       class_='ember-view l-column--grid l-column small-12 medium-6 large-4 small-valign-top l-column--equal-height')

                counter = 1
                for expand_page in expand_pages:
                    try:
                        # print("\n===========\n")
                        # print("review："+str(counter))
                        author = str(expand_page.find("span",
                                                      class_="we-truncate we-truncate--single-line ember-view we-customer-review__user").text)
                        authors.append(author)
                        # print("Author Name: " + author)

                        reviewDate = expand_page.find("time", class_="we-customer-review__date").text
                        reviewDates.append(reviewDate)
                        # print("Review Date: " + reviewDate)

                        rating = str(expand_page.find("span", class_="we-star-rating-stars-outlines").find_next()['class'])[-3]
                        ratings.append(rating)
                        # print("Reviewer Ratings: " + rating)
                        body = str(expand_page.find("div", class_="we-clamp ember-view").p.string)
                        bodies.append(body)

                        # print("Review Body: " + body )
                        counter += 1
                    except:
                        print("fail")

                reviewLib = {
                    "Author Name": authors,
                    "Review Date": reviewDates,
                    "Reviewer Ratings": ratings,
                    "Review Body": bodies
                }

                fileName = 'AppleReviewTable-{appName}.xlsx'.format(appName=appName, ratingCount=ratingCount)
                fileNameCsv = 'AppleReviewTable-{appName}.csv'.format(appName=appName, ratingCount=ratingCount)



                reviewTable = pd.DataFrame(reviewLib)
                reviewTable.to_excel(os.path.join(directory, fileName), sheet_name="Sheet1")
                reviewTable.to_csv(os.path.join(directory, fileNameCsv))

                print(reviewTable)
                driver.quit()

            summeryTableList.append(reviewTable)


        return summeryTableList








