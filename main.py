import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.edge.service import Service
from datetime import datetime
import time
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def remove_things(text):
    
    test= text.split('\n')

    # Original string parts
    month = test[2].split()[-3]
    day = int(test[2].split()[-2].split(',')[0])
    year = test[2].split()[-1]

    # Create a single string representing the date
    date_string = f"{month} {day}, {year}"

    # Parse the string into a datetime object
    date_obj = datetime.strptime(date_string, "%B %d, %Y").date()
    return test[0],test[1],date_obj,test[4]



def scrappingData(url):
    # connexion
    PATH ='edge_driver\\msedgedriver.exe'
    service = Service(PATH)
    driver = webdriver.Edge(service=service)
    url=url
    driver.get(url)
    driver.find_element(By.XPATH,"//*[@data-hook='see-all-reviews-link-foot']").click()

    elements = []
    ratings=[]
    while len(elements) < 100:
        # Find comments that start with "customer_review"
        teen_comments = driver.find_elements(By.XPATH, "//*[starts-with(@id, 'customer_review')]")
        # elements.extend(teen_comments)
        for comment in teen_comments:
            elements.append(comment.text)

        
        # teen_ratings = driver.find_elements(By.XPATH, "//*[starts-with(@id, 'customer_review')]//div[2]/a/i/span")
        # for rating in teen_ratings:
        #     ratings.append(rating.text)
        # Find ratings that start with "customer_review"
        teen_ratings = driver.find_elements(By.XPATH, "//*[starts-with(@id, 'customer_review')]//div[2]/a/i/span")
        
        # Extract ratings without using .text
        for rating in teen_ratings:
            # Using the get_attribute method to retrieve the inner HTML or a specific attribute
            ratings.append(rating.get_attribute('innerHTML'))

        # Pause briefly
        print('wsln daba l ',len(elements))
        time.sleep(1)

        # Try to find the next page button
        try:
            next_button = driver.find_element(By.XPATH, '//*[@id="cm_cr-pagination_bar"]/ul/li[2]/a')
            next_button.click()  # Click the next page button
        except NoSuchElementException:
            print("Next page button does not exist. Stopping pagination.")
            break  # Break the loop if the "Next" button is not found
        
        # Pause briefly after clicking
        time.sleep(1)
    return elements,ratings


    # turn the text to rating star to int 

def save_df(elements,ratings):
    new_ratings=[int(rating.split()[0].split('.')[0]) for rating in ratings]
    df = pd.DataFrame([remove_things(x) for x in elements], columns=['Username', 'Title', 'Date', 'Comment'])
    df['rating']=new_ratings
    df.to_csv('products.csv')
    # return df



def processing():
    df=pd.read_csv('products.csv')
    df.dropna(inplace=True)
    res = {}  # Dictionary to store the results
    sia= SentimentIntensityAnalyzer()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Comment']
        polarity_score = sia.polarity_scores(text)  # Get sentiment analysis score
        res[i] = polarity_score  # Store result in dictionary with index `i`
    sentiment_df = pd.DataFrame.from_dict(res, orient='index')
    sentiment_df=sentiment_df.join(df)  

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    # Create bar plots
    sns.barplot(data=sentiment_df, x=sentiment_df.index, y='compound', ax=ax[0])
    sns.barplot(data=sentiment_df, x=sentiment_df.index, y='pos', ax=ax[1])
    sns.barplot(data=sentiment_df, x=sentiment_df.index, y='neg', ax=ax[2])

    # Set titles for each subplot
    ax[0].set_title('Compound Sentiment')
    ax[1].set_title('Positivity')
    ax[2].set_title('Negativity')

    # Adjust the layout
    plt.tight_layout()
    plt.savefig('statistiques.png')
    # Show the plot
    plt.show()