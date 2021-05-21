import requests
import asyncio
import pandas as pd
import json
from aiohttp_requests import requests
"""
This python file is to display how we get the relative zip code for each address.
The output can be found in zip_code_crawler2.csv file.
"""

async def main():
    """
    Get zip code from google API
    :return:

    >>> main() # doctest: +ELLIPSIS
    <coroutine object main at ...
    """
    # If laptop crashes or crawling is paused by accident, paste the output to "attempt.csv" file
    # and use codes below to restart.
    """exist = pd.read_csv("attempt.csv", header=None).rename(columns={0: "zip"})
    gv = pd.read_csv("zip_code_crawler2.csv")
    gv["Full_Address"] = gv["address"] + gv["city_or_county"] + gv["state"]
    gv_il = gv[gv["date"].str.match(r"^2017.*") == True].reset_index()
    address = gv_il["Full_Address"]
    a = exist["zip"].size
    gv_il.loc[:, "zip"] = 0
    gv_il.loc[:a, "zip"] = exist["zip"]"""
    # Firstly, import the file which contains addresses of gun violence cases
    gv = pd.read_csv("zip_code_crawler1.csv", header=0, low_memory=False)
    # Add a new column for full addresses
    gv["Full_Address"] = gv["address"] + gv["city_or_county"] + gv["state"]
    address = gv["Full_Address"]
    # Add a column for zip codes
    gv["zip"] = ""
    k = 0
    for i in address.to_list():
        if gv["zip"][k] == "":
            # Request geolocation data from google API
            site = await requests.get("https://maps.googleapis.com/maps/api/geocode/json?address="+str(i)+"&key=AIzaSyCQO-C--gDlOWZ8xW2xNhzrGNHISXWgwQ4")
            detail = await site.json()
            # Extract zip codes from pages
            if detail["results"]:
                address1 = detail["results"][0]['address_components']
                zip = None
                for x in address1:
                    if x["types"] == ['postal_code']:
                        zip = x["long_name"]
            else:
                zip = None
            gv.iloc[k, -1] = zip
            print(zip)
        k += 1
        # Store the results into new file per 10000 requests. 
        if k % 10000 == 0:
            gv.to_csv("zip_code_crawler2.csv")
    # Store the results into new file after all requests are done 
    gv.to_csv("zip_code_crawler2.csv")

if __name__ == '__main__':
    asyncio.run(main())
