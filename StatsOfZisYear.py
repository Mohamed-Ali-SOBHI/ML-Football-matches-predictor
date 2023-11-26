from time import sleep
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import os
import sys


def get_soup(url):
    """
    Get the soup from the url
    :param url: url to get the soup from
    :return: soup
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def get_data_from_soup(soup,seasons,leagues,baseUrl):
    """
    Get the data from the soup
    :param soup: soup to get the data from
    :return: data
    """
    data = {}

    for league in tqdm(range(len(leagues))):
        for season in range(len(seasons)):
            url = baseUrl + '/' + leagues[league] + '/' + seasons[season] # get url for eache league and season
            soup = get_soup(url) # get the soup from the url
            script = soup.find_all('script')[2].string # get the script tags
            stringsInJson = script.split("('")[1].split("')") # get the string in the script tags
            jsonData = stringsInJson[0].encode('utf-8').decode('unicode_escape') 
            data[leagues[league] + ' ' + seasons[season]] = json.loads(jsonData) # get the json data from the string
                   
    return data


def extract_stats_from_data(data):
    """
    Extract the stats from the data
    :param data: data to extract the stats from
    :return: stats
    """
    stats = {}

    for season in data.keys():
        for team in data[season].keys():
            teamName = data[season][team]['title']
            stats[season + ' ' + teamName] = data[season][team]['history']
    return stats

def save_data(stats):
    """
    Save the stats to a csv file
    :param stats: stats to save
    """
    # download statistics in csv format for each team in each season
    for team in stats.keys():
        dir = team.split(' ')[0]
        #split the team name from the index 2 to the end of the string 
        teamName = team.split(' ')[2:]
        season = '2023'
        
        # Define the base directory
        base_dir = 'ML-Football-matches-predictor/StatsOfZisYear/' + dir
        
        # Check if the directory exists. If not, create it.
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Depending on the league, save to the corresponding directory
        if dir == 'La_liga':
            df = pd.DataFrame(stats[team])
            df.to_csv(base_dir + '/' + season + ' ' + ' '.join(teamName) + '.csv', index=False)
        elif dir == 'Bundesliga':
            df = pd.DataFrame(stats[team])
            df.to_csv(base_dir + '/' + season + ' ' + ' '.join(teamName) + '.csv', index=False)
        elif dir == 'EPL':
            df = pd.DataFrame(stats[team])
            df.to_csv(base_dir + '/' + season + ' ' + ' '.join(teamName) + '.csv', index=False)
        elif dir == 'Serie_A':
            df = pd.DataFrame(stats[team])
            df.to_csv(base_dir + '/' + season + ' ' + ' '.join(teamName) + '.csv', index=False)
        elif dir == 'Ligue_1':
            df = pd.DataFrame(stats[team])
            df.to_csv(base_dir + '/' + season + ' ' + ' '.join(teamName) + '.csv', index=False)
        elif dir == 'RFPL':
            df = pd.DataFrame(stats[team])
            df.to_csv(base_dir + '/' + season + ' ' + ' '.join(teamName) + '.csv', index=False)
        else:
            print('Error: ' + dir + ' is not a valid league')

if __name__ == '__main__':
    
    baseUrl = 'https://understat.com/league'
    leagues = ['La_liga', 'Bundesliga', 'EPL','Serie_A', 'Ligue_1',"RFPL"]
    seasons = '2023'
    
    soup = get_soup(baseUrl)
    data = get_data_from_soup(soup,seasons,leagues,baseUrl)
    stats = extract_stats_from_data(data)
    save_data(stats)