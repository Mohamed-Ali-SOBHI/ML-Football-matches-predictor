from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv
import os


def scrappe(urls):
    """
    Scrappe the data from the url
    :param url: url to scrappe the data from
    :return: data
    """
    
    matches_data = []

    for url in urls:
        # Configuration du navigateur pour le mode headless
        options = webdriver.ChromeOptions()
        options.headless = True

        # Ouvrir le navigateur et accéder à la page
        driver = webdriver.Chrome(options=options)  # Assurez-vous que ChromeDriver est dans votre PATH
        print('processing url:', url)
        driver.get(url)

        try:
            # Utiliser une attente explicite pour attendre que les matchs soient chargés
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'grid-event-wrapper'))
            WebDriverWait(driver, 20).until(element_present)

            # Obtenir le contenu de la page
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Trouver tous les matchs
            matches = soup.find_all('div', class_='grid-event-wrapper')
            
            for match in matches:
                    teams = match.find_all('div', class_='participant')
                    if teams:
                        team_1 = teams[0].get_text(strip=True)
                        team_2 = teams[1].get_text(strip=True) if len(teams) > 1 else None
                    else:
                        team_1, team_2 = None, None

                    # Cotes
                    odds = match.find_all('div', class_='option-value')
                    if odds:
                        odd_1 = odds[0].get_text(strip=True)
                        odd_2 = odds[1].get_text(strip=True) if len(odds) > 1 else None
                        odd_3 = odds[2].get_text(strip=True) if len(odds) > 2 else None
                    else:
                        odd_1, odd_2, odd_3 = None, None, None

                    match_data = {
                        'team_1': team_1,
                        'team_2': team_2,
                        'odd_1': odd_1,
                        'odd_2': odd_2,
                        'odd_3': odd_3,
                    }
                    matches_data.append(match_data)
                    
        finally:
            driver.quit()   

    return matches_data

def save_data(matches_data, day):
    """
    Save the matches data to a csv file
    :param matches_data: matches data to save
    :param day: string indicating the day (e.g. "today" or "tomorrow" or "plus_2")
    """
    if day == "today":
        date_str = datetime.now().strftime('%Y-%m-%d')
    elif day == "tomorrow":
        date_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # ajouter un jour
    elif day == "plus_2":
        date_str = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    
    directory = "ML-Football-matches-predictor/MatchOfZeDay"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/matches_of_ze_day_{date_str}.csv"
    
    keys = matches_data[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(matches_data)

if __name__ == '__main__':
    
    urls_aujourdhui = [
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/espagne-28/laliga-102829", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd'hui/angleterre-14/premier-league-102841", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/france-16/ligue-1-102843", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/allemagne-17/bundesliga-102842", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/italie-20/serie-a-102846",
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/russie-25/premier-league-102850"
    ]

    # Pour aujourd'hui
    matches_data_today = scrappe(urls_aujourdhui)
    print(matches_data_today)
    save_data(matches_data_today, "today")
    
    # Pour demain
    urls_demain = [
        "https://sports.bwin.fr/fr/sports/football-4/demain/espagne-28/laliga-102829", 
        "https://sports.bwin.fr/fr/sports/football-4/demain/angleterre-14/premier-league-102841", 
        "https://sports.bwin.fr/fr/sports/football-4/demain/france-16/ligue-1-102843", 
        "https://sports.bwin.fr/fr/sports/football-4/demain/allemagne-17/bundesliga-102842", 
        "https://sports.bwin.fr/fr/sports/football-4/demain/italie-20/serie-a-102846",
        "https://sports.bwin.fr/fr/sports/football-4/demain/russie-25/premier-league-102850"
    ]
    matches_data_tomorrow = scrappe(urls_demain)
    print(matches_data_tomorrow)
    save_data(matches_data_tomorrow, "tomorrow")
    
    # Pour après-demain
    url_apres_2_jours = [ 
        "https://sports.bwin.fr/fr/sports/football-4/après-2-jours/espagne-28/laliga-102829", 
        "https://sports.bwin.fr/fr/sports/football-4/après-2-jours/angleterre-14/premier-league-102841",
        "https://sports.bwin.fr/fr/sports/football-4/après-2-jours/france-16/ligue-1-102843", 
        "https://sports.bwin.fr/fr/sports/football-4/après-2-jours/allemagne-17/bundesliga-102842", 
        "https://sports.bwin.fr/fr/sports/football-4/après-2-jours/italie-20/serie-a-102846",
        "https://sports.bwin.fr/fr/sports/football-4/après-2-jours/russie-25/premier-league-102850"
    ]

    matches_data_plus_2 = scrappe(url_apres_2_jours)
    print(matches_data_plus_2)
    save_data(matches_data_plus_2, "plus_2")
