from datetime import datetime, timedelta
import json
import os
import time
import numpy as np  
import pandas as pd 
import requests  
from bs4 import BeautifulSoup
from tqdm import tqdm  

def get_soup(url):
    """
    Send a GET request to the specified URL and return a BeautifulSoup object
    that can be used to extract data from the HTML content of the response.

    Parameters:
    url (str): The URL to send the GET request to.

    Returns:
    BeautifulSoup: The BeautifulSoup object created from the HTML content of the response.
    """
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup

def get_data(soup, date):
    """
    Extract the data of the matches of the day from the BeautifulSoup object
    and store it in a pandas DataFrame.

    Parameters:
    soup (BeautifulSoup): The BeautifulSoup object to extract the data from.

    Returns:
    pandas.DataFrame: The pandas DataFrame containing the data of the matches of the day.
    """
    table_date = soup.find_all('div', {'class': 'date bold'}) #get the dates of the matches
    
    # Find the index of today's date in the list of dates
    for i in range(len(table_date)):
        table_date[i] = table_date[i].text.strip()[0:8]
        if table_date[i] == date:
            index = i
            break
        
    table = soup.find_all('table')[index] #get the table containing the matches of the day
    trhead = table.find('tr', class_='head') #get the header of the table
    champ = trhead.find_all('td')[0].text #get the name of the championship
    equipe1 = trhead.find_all('td')[4].text #get the name of the first team
    cote = 'cote' 
    equipe2 = trhead.find_all('td')[6].text #get the name of the second team

    MatchOfZeDay = pd.DataFrame(columns=[champ, equipe1, cote, equipe2]) #create a DataFrame to store the data
    tcontent = table.find_all('tr', class_='tr-border-green') #get the rows of the table
    # For each row of the table get the name of the championship, the names of the teams and the odds
    for td in tcontent:
        champ = td.find_all('td')[0].text
        equipe1 = td.findAll('td')[4].text
        cotetd = td.findAll('td')[5]
        cotediv = cotetd.find('div', class_='flex-content')
        divs = cotediv.find_all('div')
        # Get the odds
        cote = []
        for strong in divs:
            cote.append(strong.text)
        equipe2 = td.findAll('td')[6].text
        MatchOfZeDay.loc[len(MatchOfZeDay)] = [champ, equipe1, cote, equipe2]
    return MatchOfZeDay


def clean_data(MatchOfZeDay):
    """
    Clean the DataFrame by removing the rows that don't have odds.

    Parameters:
    MatchOfZeDay (pandas.DataFrame): The DataFrame to clean.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    MatchOfZeDay["cote"] = MatchOfZeDay["cote"].apply(lambda x: [i for i in x if i != '']) #remove the empty strings
    MatchOfZeDay["cote"] = MatchOfZeDay["cote"].apply(lambda x: [i for i in x if i != '-']) #remove the '-' strings
    MatchOfZeDay = MatchOfZeDay[MatchOfZeDay["cote"].map(len) > 0] #remove the rows that don't have odds
    #keep only the european championships "AL1", "ANP", "FR1", "IT1", "ES1"
    MatchOfZeDay = MatchOfZeDay[MatchOfZeDay["CHAMP"].isin(["AL1", "ANP", "FR1", "IT1", "ES1"])]
    return MatchOfZeDay

def store_data(MatchOfZeDay, date):
    """
    Store the DataFrame in a CSV file.

    Parameters:
    MatchOfZeDay (pandas.DataFrame): The DataFrame to store.
    """
    dir = 'ML-Football-matches-predictor\MatchOfZeDay'
    #format the date to %d-%m-%y
    date = date[0:2] + '-' + date[3:5] + '-' + date[6:8]
    MatchOfZeDay.to_csv(f"{dir}/matches_of_ze_day_{date}.csv", index=False)
    
    
if __name__ == '__main__':
    
    url = 'https://mdjs.ma/resultats/programme-cote-sport.html'
    
    soup = get_soup(url)
    today = time.strftime("%d/%m/%Y") #get today's date
    today = today[0:6] + today[8:10] #format the date to match the format of the dates in the HTML
    MatchOfZeDay = get_data(soup, today)
    MatchOfZeDay = clean_data(MatchOfZeDay)
    store_data(MatchOfZeDay, today)
    
    soup = get_soup(url)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%d/%m/%Y') #get tomorrow's date
    tomorrow = tomorrow[0:6] + tomorrow[8:10] #format the date to match the format of the dates in the HTML
    MatchOfZeDay = get_data(soup, tomorrow)
    MatchOfZeDay = clean_data(MatchOfZeDay)
    store_data(MatchOfZeDay, tomorrow)
    
    soup = get_soup(url)
    plus_2 = (datetime.now() + timedelta(days=2)).strftime('%d/%m/%Y') #get the date after tomorrow
    plus_2 = plus_2[0:6] + plus_2[8:10] #format the date to match the format of the dates in the HTML
    MatchOfZeDay = get_data(soup, plus_2)
    MatchOfZeDay = clean_data(MatchOfZeDay)
    store_data(MatchOfZeDay, plus_2)
    """
    soup = get_soup(url)
    plus_3 = (datetime.now() + timedelta(days=3)).strftime('%d/%m/%Y') #get the date after tomorrow
    plus_3 = plus_3[0:6] + plus_3[8:10] #format the date to match the format of the dates in the HTML
    MatchOfZeDay = get_data(soup, plus_3)
    MatchOfZeDay = clean_data(MatchOfZeDay)
    store_data(MatchOfZeDay, plus_3)
    """