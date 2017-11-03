CITS4404 Group C1

To run the Twitter Scraper, you need to create a 'config.yml file'. You will need to register
a Twitter App to get the required Keys and Access Tokens.

The file should contain:

<start_file>
twitter:
  consumer_key: CONSUMER_KEY
  consumer_secret: CONSUMER_SECRET
  access_token: ACCESS_TOKEN
  access_token_secret: ACCESS_TOKEN_SECRET
<end_file>

Running steps:

1. Run main.py until you are satisfied with the amount of data.

2. Run clean_data.py to clean the data.