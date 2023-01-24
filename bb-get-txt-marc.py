# NEED TO TURN INTO FUNCTION
# trying to get all txt and marcxml files for all items in banned books collection
# documentation- https://archive.org/services/docs/api/internetarchive/api.html#internetarchive.download
# https://archive.org/services/docs/api/internetarchive/quickstart.html#metadata
# 

import internetarchive as ia
from internetarchive import download
from internetarchive import get_files

search = ia.search_items('collection:bannedbooks')
# https://archive.org/details/bannedbooks

id_list = []

for result in search:
    id_list.append(result['identifier'])

# from last week I know that the code works up to here.
counter = 0

for item in id_list:
    # formats we want are MARC and txt, djvu txt preserves pdf 'look' I think
    # change destdir to D drive because I don't have space on C, NEED TO CHANGE BACK TO D
    # ignore errors because some links to files are broken and it'll terminate if a single file fails to download otherwise
    # item_index = item PROB WONT WORK because item is apparently a string
    # checksum I think this also helps avoid duplicates
        # only want a few items for testng
        # will len work? 
    if len(list(get_files(item, formats=['DjVuTXT', 'MARC']))) >= 2:
        download(
            item, formats=['DjVuTXT','MARC'], 
            destdir=r'D:\ML-BannedBooks-Test', 
            ignore_errors=True, checksum=True
            )
        counter += 1
    if counter >= 50:
        break

# the goal is to get a list of file types and only download if that list contains both djvutxt and marc
# beyond that, I think my if statement containing get_files will work as long as len works on whatever type of object the get_files function returns
# but I'm really getting over my head here
# info on download and get_files syntax can be found at:
# https://archive.org/developers/internetarchive/api.html#downloading
# https://archive.org/developers/internetarchive/api.html#file-objects  