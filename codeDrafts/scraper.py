# Import the beautifulsoup 
# and request libraries of python.
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

def find_most_common_word(strings):
    # Define a list of words and patterns to exclude
    excluded_words = ["pill", "drug", "images", "tablet", "rectangle", ".com"]
    excluded_pattern = re.compile(r'[.,!?():;"]')

    # Create a list to store cleaned words
    cleaned_words = []

    # Split each string into words, clean them, and add to the list
    for string in strings:
        words = string.split()
        cleaned_words.extend([word.lower() for word in words if word.lower() not in excluded_words])

    # Remove punctuation and excluded patterns from cleaned words
    cleaned_words = [re.sub(excluded_pattern, '', word) for word in cleaned_words]

    # Filter out words with less than 5 letters
    cleaned_words = [word for word in cleaned_words if len(word) >= 5]

    # Count the occurrences of each cleaned word
    word_counts = Counter(cleaned_words)

    # Find the most common word
    most_common_word = word_counts.most_common(5)

    if most_common_word:
        return most_common_word
    else:
        return None
    
def search_drug_by_imprint(imprint):
    search_query = "What drug is the pill imprint:" + imprint
    url = f"https://www.google.com/search?q={search_query}"
    
    try:
        request_result = requests.get(url)
        # response.raise_for_status()
        
        soup = BeautifulSoup(request_result.text,
                                "html.parser")
        
        heading_object = soup.find_all( 'h3' )

        search_results = []
        for result in heading_object:
            search_results.append(result.getText())
        
        # Return up to 5 drug names
        return find_most_common_word(search_results)
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    imprint = input("Enter the pill imprint: ")
    search_results = search_drug_by_imprint(imprint)
    
    if search_results:
        for result in search_results:
            print(result)

    #     print("Possible drugs:")
    #     for result in search_results:
    #         print(result)
    #     most_common = find_most_common_word(search_results)
    #     print("Most common:")
    #     print(most_common)
    
    else:
        print("No drugs found for the given imprint.")

# def get_headlines(text):
#         # Combine two strings with default google search URL
#         # 'https://google.com/search?q=' and
#         # our customized search keyword parameter.
#         # Concatenate them
#         url = 'https://google.com/search?q=' + text

#         # Fetch the URL data using requests.get(url),
#         # store it in a variable, request_result.
#         request_result=requests.get( url )
        
#         # Creating soup from the fetched request
#         soup = bs4.BeautifulSoup(request_result.text,
#                                 "html.parser")
#         # soup.find.all( h3 ) to grab 
#         # all major headings of our search result,
#         heading_object=soup.find_all( 'h3' )
        
#         # Iterate through the object 
#         # and add it to the list
#         headlines = []
#         for info in heading_object:
#             headlines.append(info.getText())

#         return headlines

# pittsburgh_headlines = get_headlines("pittsburgh")
# for text in pittsburgh_headlines:
#     print(text)