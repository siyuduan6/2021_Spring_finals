import requests
import asyncio
import pandas as pd
import json
from aiohttp_requests import requests

async def main():
    """exist = pd.read_csv("attempt1.txt", header=None).rename(columns={0: "zip"})
    gv = pd.read_csv("stage3.csv")
    gv["Full_Address"] = gv["address"] + gv["city_or_county"] + gv["state"]
    gv_il = gv[gv["date"].str.match(r"^2017.*") == True].reset_index()
    address = gv_il["Full_Address"]
    a = exist["zip"].size
    gv_il.loc[:, "zip"] = 0
    gv_il.loc[:a, "zip"] = exist["zip"]"""
    gv = pd.read_csv("zip_code_crawler.csv", header=0)
    gv["Full_Address"] = gv["address"] + gv["city_or_county"] + gv["state"]
    address = gv["Full_Address"]
    k = 0
    for i in address.to_list():
        if gv["zip"][k] == "":
            site = await requests.get("https://maps.googleapis.com/maps/api/geocode/json?address="+str(i)+"&key=AIzaSyCQO-C--gDlOWZ8xW2xNhzrGNHISXWgwQ4")
            detail = await site.json()
            if detail["results"]:
                address1 = detail["results"][0]['address_components']
                zip = None
                for x in address1:
                    if x["types"] == ['postal_code']:
                        zip = x["long_name"]
            else:
                zip = None
            gv.iloc[k, -1] = zip
            print(zip, k)
        k += 1
        if k % 10000 == 0:
            gv.to_csv("zip_code_crawler2.csv")
            print(gv)
    gv.to_csv("zip_code_crawler2.csv")

if __name__ == '__main__':
    asyncio.run(main())





