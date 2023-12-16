import requests
from bs4 import BeautifulSoup

url = "https://gomocup.org/results/gomocup-result-2019/"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Assuming the game data is stored in a specific HTML element
    game_data_elements = soup.find_all("div", class_="your-class-name")  # Replace with the actual class name

    # Process the game data elements and extract the relevant information
    for game_data_element in game_data_elements:
        # Extract and process the data (adjust as needed)
        data = game_data_element.text
        print(data)
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
