

# Import necessary libraries for data handling, web scraping, sentiment analysis, and visualization
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.edge.service import Service
from datetime import datetime
import time
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from wordcloud import WordCloud

def remove_things(text):
    # Split text by newline to separate fields
    test = text.split('\n')

    # Extract date parts from text
    month = test[2].split()[-3]
    day = int(test[2].split()[-2].split(',')[0])
    year = test[2].split()[-1]

    # Combine parts into a date string
    date_string = f"{month} {day}, {year}"

    # Convert string to date object
    date_obj = datetime.strptime(date_string, "%B %d, %Y").date()
    return test[0], test[1], date_obj, test[4]

def scrappingData(url):
    elements = []  # List to store scraped reviews
    ratings = []  # List to store ratings associated with reviews
    Product_link = None  # Placeholder for product image link

    try:
        # Set up the web driver with the specified path to the Edge driver
        PATH = 'edge_driver\\msedgedriver.exe'
        service = Service(PATH)
        driver = webdriver.Edge(service=service)

        # Open the product page URL
        driver.get(url)
        time.sleep(10)  # Wait for the page to fully load

        # Try to locate the product image and get its link, if available
        try:
            landing_image = driver.find_element(By.XPATH, '//*[@id="landingImage"]')
            Product_link = landing_image.get_attribute("src")
            print("Product Image Link:", Product_link)
        except NoSuchElementException:
            print("Product image not found.")

        # Attempt to click on the "See all reviews" link, if it exists
        try:
            driver.find_element(By.XPATH, "//*[@data-hook='see-all-reviews-link-foot']").click()
        except NoSuchElementException:
            print("See all reviews link not found.")
            driver.close()
            return elements, ratings, Product_link

        # Scrape reviews and ratings until the specified number is reached
        while len(elements) < 100:
            try:
                # Attempt to extract customer reviews if present
                teen_comments = driver.find_elements(By.XPATH, "//*[starts-with(@id, 'customer_review')]")
                if teen_comments:
                    for comment in teen_comments:
                        elements.append(comment.text)
                
                # Attempt to extract ratings associated with reviews if present
                teen_ratings = driver.find_elements(By.XPATH, "//*[starts-with(@id, 'customer_review')]//div[2]/a/i/span")
                if teen_ratings:
                    for rating in teen_ratings:
                        ratings.append(rating.get_attribute('innerHTML'))
                
                print('Scraped reviews so far:', len(elements))
                time.sleep(1)

                # Attempt to navigate to the next page of reviews, if available
                try:
                    next_button = driver.find_element(By.XPATH, '//*[@id="cm_cr-pagination_bar"]/ul/li[2]/a')
                    next_button.click()
                except NoSuchElementException:
                    print("Next page button does not exist. Stopping pagination.")
                    break

                time.sleep(1)

            except Exception as e:
                print(f"Error while scraping reviews: {e}")
                break

    except WebDriverException as e:
        print(f"Error initializing web driver or accessing the URL: {e}")
    
    except TimeoutException as e:
        print(f"Page load timed out: {e}")
    
    finally:
        # Close the browser window
        driver.close()

    return elements, ratings, Product_link

def get_product_name(url):
    # Extract product name from the URL by splitting it
    return url.split('/')[3]


def save_df(elements, ratings, url):
    # Convert ratings to integers
    new_ratings = [int(rating.split()[0].split('.')[0]) for rating in ratings]

    # Create a DataFrame from reviews and their details
    df = pd.DataFrame([remove_things(x) for x in elements], columns=['Username', 'Title', 'Date', 'Comment'])
    df['rating'] = new_ratings

    # Save DataFrame to CSV
    product_name = get_product_name(url)
    df.to_csv(f'./static/data/{product_name}.csv', index=False)
    return product_name


def polarity_score_raberta(example):
    # Load tokenizer and model for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Encode text and obtain model output
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)  # Apply softmax to get probabilities

    # Store scores in a dictionary
    score_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
    }
    return score_dict


def polarity_score(df):
    res = {}  # Dictionary to store the results

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Comment']
        res[i] = polarity_score_raberta(text)  # Get sentiment analysis score
    return res


def positivity(a, b, c):
    # Determine sentiment based on the highest score
    if a >= b and a >= c:
        return 'positif'
    elif b >= a and b >= c:
        return 'neutral'
    return 'negatif'


def processing(product_name):
    # Load the saved DataFrame of reviews
    df = pd.read_csv(f'./static/data/{product_name}.csv')
    df.dropna(inplace=True)

    # Perform sentiment analysis on each review
    res = polarity_score(df)
    sentiment_df = pd.DataFrame.from_dict(res, orient='index')
    sentiment_df = sentiment_df.join(df)
    sentiment_df['sentiment'] = sentiment_df.apply(lambda row: positivity(row['roberta_pos'], row['roberta_neu'], row['roberta_neg']), axis=1)

    # Calculate percentages of each sentiment type
    negative = len(sentiment_df[sentiment_df['sentiment'] == 'negatif']) / sentiment_df.shape[0] * 100
    neutral = len(sentiment_df[sentiment_df['sentiment'] == 'neutral']) / sentiment_df.shape[0] * 100
    positif = len(sentiment_df[sentiment_df['sentiment'] == 'positif']) / sentiment_df.shape[0] * 100

    # Plot bar chart of sentiment distribution
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'])
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig(f'./static/images/plots/bar{product_name}.png')

    # Plot pie chart of sentiment distribution
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'orange', 'red'], startangle=90)
    plt.title('Sentiment Distribution')
    plt.ylabel('')
    plt.savefig(f'./static/images/plots/Pie{product_name}.png')

    # Generate word cloud from review comments
    text = ' '.join(df['Comment'].astype(str))
    wordcloud = WordCloud(background_color='white', max_words=100).generate(text)
    plt.figure(figsize=(6, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f'./static/images/plots/wordcloud{product_name}.png')

    # Plot correlation heatmap for sentiment scores
    plt.figure(figsize=(6, 4))
    sns.heatmap(sentiment_df[['roberta_neg', 'roberta_neu', 'roberta_pos']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Sentiment Probabilities')
    plt.savefig(f'./static/images/plots/Correlation{product_name}.png')

    # Plot sentiment trends over time
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_by_date = sentiment_df.groupby(sentiment_df['Date'].dt.date)['sentiment'].value_counts().unstack()
    sentiment_by_date.plot(kind='line', figsize=(6, 4), color=['green', 'orange', 'red'])
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.savefig(f'./static/images/plots/line{product_name}.png')

    return positif, neutral, negative
