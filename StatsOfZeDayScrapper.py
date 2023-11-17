import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time

def scrape(leagues):
    """
    Scrape for xG, xGA, and xPts for each team in the top 5
    European Leagues.
    :param leagues: list of leagues to scrape
    :return: list of dicts containing league, teams, xG, xGA, and xPts
    """
    league_data = []

    for league in tqdm(leagues):
        print('processing league:', league)
        teams, stats = extract_data(league, url) # extract data from website
        num_teams = get_num_teams(league) # get number of teams in league
        xG_range, xGA_range, xPts_range = calculate_index_ranges(num_teams) # calculate index ranges for xG, xGA, and xPts
        xG, xGA, xPts = extract_stats(stats, xG_range, xGA_range, xPts_range) # extract xG, xGA, and xPts
        print(xG, xGA, xPts)
        league_dict = create_league_dict(league, teams, xG, xGA, xPts) # create dict for league
        league_data.append(league_dict) # append dict to league_data
        save_to_csv(league, league_dict) # save league data to csv

    return league_data

def extract_data(league,url):
    """
    Extract data from website
    :param league: league to extract data from
    :param url: website url
    :return: list of teams and list of stats
    """
    response = requests.get(url  + league)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find_all('div', class_='ranking-row') # find table containing data

    teams = []
    stats = []
    for i in range(len(table)):
        teams.append(table[i].find_all('div', class_='main')) # extract team names
        cells = table[i].find_all('div', class_='cells') # extract stats 
        stat = cells[0].find_all('div', class_='cell')[3].text # get the 4th stat
        stats.append(stat)
        

    for i in range(len(teams)):
        if len(teams[i]) == 0: # skip the first row of the table
            continue
        teams[i] = teams[i][0].text # extract team name

    return teams, stats

def get_num_teams(league):
    """
    Get number of teams in league
    :param league: league to get number of teams from
    :return: number of teams in league
    """
    num_teams = 20
    if league == 'bundesliga':
        num_teams = 16
    return num_teams

def calculate_index_ranges(num_teams):
    """
    Calculate the index ranges for xG, xGA, and xPts
    :param num_teams: number of teams in league
    :return: xG_range, xGA_range, xPts_range
    """
    xG_range = range(1, num_teams + 1) # xG range [1, 21 or 19 if bundesliga]
    xGA_range = range(num_teams + 2, (num_teams * 2) + 2) # xGA range [22, 42 or 40 if bundesliga]
    xPts_range = range((num_teams * 2) + 3, (num_teams * 3) + 3) # xPts range [43, 63 or 61 if bundesliga]
    #print(xG_range, xGA_range, xPts_range)
    return xG_range, xGA_range, xPts_range

def extract_stats(stats, xG_range, xGA_range, xPts_range):
    """
    Extract xG, xGA, and xPts
    :param stats: list of stats
    :param xG_range: index range for xG
    :param xGA_range: index range for xGA
    :param xPts_range: index range for xPts
    :return: xG, xGA, xPts
    """
    xG = [stats[i] for i in xG_range] # extract xG from stats list getted from extract_data
    xGA = [stats[i] for i in xGA_range] # extract xGA from stats list getted from extract_data
    xPts = [stats[i] for i in xPts_range] # extract xPts from stats list getted from extract_data
    return xG, xGA, xPts

def create_league_dict(league, teams, xG, xGA, xPts):
    """
    Create dict for league
    :param league: league to create dict for
    :param teams: list of teams
    :param xG: list of xG
    :param xGA: list of xGA
    :param xPts: list of xPts
    :return: dict containing league, teams, xG, xGA, xPts
    """
    num_teams = get_num_teams(league) 
    return {'league': league , 'teams': teams[1:num_teams+1], 'xG': xG, 'xGA': xGA, 'xPts': xPts} 
    
def save_to_csv(league, league_dict):
    """
    Save league data to csv
    :param league: league to save data for
    :param league_dict: dict containing league, teams, xG, xGA, and xPts
    """
    df = pd.DataFrame.from_dict(league_dict) #convert dict to dataframe
    today = time.strftime("%d-%m-%Y") # get today's date
    df.to_csv('ML-Football-matches-predictor\StatsOfZeDay/' + today + ' ' + league + '.csv') # save dataframe to csv in Prod\StatsOfZeDay folder


if __name__ == '__main__':
    leagues = ['bundesliga', 'premier-league', 'serie-a', 'ligue-1', 'la-liga'] # list of leagues to scrape
    url = 'https://oddalerts.com/xg/'
    league_data = scrape(leagues) 
    print(league_data)