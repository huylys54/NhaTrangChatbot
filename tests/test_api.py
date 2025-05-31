import googlemaps
from langchain_community.utilities import GooglePlacesAPIWrapper
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

# gmaps = googlemaps.Client(key=os.getenv('GOOGLEMAP_API_KEY'))

# search_query = f"Vinpearl land"

# results = gmaps.find_place(
#     input=search_query,
#     input_type="textquery",
#     fields=["geometry", "name", "formatted_address", "place_id"]
# )
# candidates = results.get("candidates", [])
# print(candidates)
# if not candidates:
#     print("No candidates found for the search query.")
# place = candidates[0]
# place_id = place["place_id"]
# map_url = f"https://www.google.com/maps/search/?api=1&query=Google&query_place_id={place_id}"
# context = (
#     f"Location: {place['name']}\n"
#     f"Address: {place['formatted_address']}\n"
#     f"Map: {map_url}"
# )

# print(context)

gmaps = GooglePlacesAPIWrapper(top_k_results=5)

search_query = f"quán ăn ngon tại nha trang"
results = gmaps.run(search_query)
# The wrapper returns a string summary, but you can also parse for place_id if needed
# For a map link, you can use Google Maps search URL
map_url = f"https://www.google.com/maps/search/?api=1&query={search_query.replace(' ', '+')}"
context = f"{results}\nMap: {map_url}"

print(context)
