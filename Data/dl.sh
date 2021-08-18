#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nrGb7eEQj7h5YC5kv65hcrtIhr7l03AI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nrGb7eEQj7h5YC5kv65hcrtIhr7l03AI" -O DBLP.zip && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15gdrbv9l7luEFHq3YdAkeWYldHcqEjG9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15gdrbv9l7luEFHq3YdAkeWYldHcqEjG9" -O Freebase.zip && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZEi2sTaZ2bk8cQwyCxtlwuWJsAq9N-Cl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZEi2sTaZ2bk8cQwyCxtlwuWJsAq9N-Cl" -O PubMed.zip && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pS78jCtnAnlAQfcfJMkWnJ2utInDu91k' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pS78jCtnAnlAQfcfJMkWnJ2utInDu91k" -O Yelp.zip && rm -rf /tmp/cookies.txt

