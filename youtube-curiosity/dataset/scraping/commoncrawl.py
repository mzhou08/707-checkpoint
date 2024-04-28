# Gets all YouTube videos from CommonCrawl

import time
import boto3
import dotenv
import os
import json

dotenv.load_dotenv("../../.env")

ALL_CRAWLS = ['CC-MAIN-2023-50', 'CC-MAIN-2023-40', 'CC-MAIN-2023-23', 'CC-MAIN-2023-14', 'CC-MAIN-2023-06', 'CC-MAIN-2022-49', 'CC-MAIN-2022-40', 'CC-MAIN-2022-33', 'CC-MAIN-2022-27', 'CC-MAIN-2022-21', 'CC-MAIN-2022-05', 'CC-MAIN-2021-49', 'CC-MAIN-2021-43', 'CC-MAIN-2021-39', 'CC-MAIN-2021-31', 'CC-MAIN-2021-25', 'CC-MAIN-2021-21', 'CC-MAIN-2021-17', 'CC-MAIN-2021-10', 'CC-MAIN-2021-04', 'CC-MAIN-2020-50', 'CC-MAIN-2020-45', 'CC-MAIN-2020-40', 'CC-MAIN-2020-34', 'CC-MAIN-2020-29', 'CC-MAIN-2020-24', 'CC-MAIN-2020-16', 'CC-MAIN-2020-10', 'CC-MAIN-2020-05', 'CC-MAIN-2019-51', 'CC-MAIN-2019-47', 'CC-MAIN-2019-43', 'CC-MAIN-2019-39', 'CC-MAIN-2019-35', 'CC-MAIN-2019-30', 'CC-MAIN-2019-26', 'CC-MAIN-2019-22', 'CC-MAIN-2019-18', 'CC-MAIN-2019-13', 'CC-MAIN-2019-09', 'CC-MAIN-2019-04', 'CC-MAIN-2018-51', 'CC-MAIN-2018-47', 'CC-MAIN-2018-43', 'CC-MAIN-2018-39', 'CC-MAIN-2018-34', 'CC-MAIN-2018-30', 'CC-MAIN-2018-26', 'CC-MAIN-2018-22', 'CC-MAIN-2018-17', 'CC-MAIN-2018-13', 'CC-MAIN-2018-09', 'CC-MAIN-2018-05', 'CC-MAIN-2017-51', 'CC-MAIN-2017-47', 'CC-MAIN-2017-43', 'CC-MAIN-2017-39', 'CC-MAIN-2017-34', 'CC-MAIN-2017-30', 'CC-MAIN-2017-26', 'CC-MAIN-2017-22', 'CC-MAIN-2017-17', 'CC-MAIN-2017-13', 'CC-MAIN-2017-09', 'CC-MAIN-2017-04', 'CC-MAIN-2016-50', 'CC-MAIN-2016-44', 'CC-MAIN-2016-40', 'CC-MAIN-2016-36', 'CC-MAIN-2016-30', 'CC-MAIN-2016-26', 'CC-MAIN-2016-22', 'CC-MAIN-2016-18', 'CC-MAIN-2016-07', 'CC-MAIN-2015-48', 'CC-MAIN-2015-40', 'CC-MAIN-2015-35', 'CC-MAIN-2015-32', 'CC-MAIN-2015-27', 'CC-MAIN-2015-22', 'CC-MAIN-2015-18', 'CC-MAIN-2015-14', 'CC-MAIN-2015-11', 'CC-MAIN-2015-06', 'CC-MAIN-2014-52', 'CC-MAIN-2014-49', 'CC-MAIN-2014-42', 'CC-MAIN-2014-41', 'CC-MAIN-2014-35', 'CC-MAIN-2014-23', 'CC-MAIN-2014-15', 'CC-MAIN-2014-10', 'CC-MAIN-2013-48', 'CC-MAIN-2013-20', 'CC-MAIN-2009-2010', 'CC-MAIN-2008-2009']
# ALL_CRAWLS = ['CC-MAIN-2023-50']

MAX_TIMEOUT_S = 20

YOUTUBE_VIDEOS_OUTPUT_FILE = "/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/commoncrawl_videos.json"
YOUTUBE_CHANNELS_OUTPUT_FILE = "/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/commoncrawl_channels.json"

client = boto3.client(
    'athena',
    region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)

# refresh ccindex table with latest crawls
client.start_query_execution(
    QueryString = 'MSCK REPAIR TABLE ccindex',
    QueryExecutionContext = {
        'Database': 'ccindex'
    },
    ResultConfiguration = { 'OutputLocation': os.environ['AWS_S3_BUCKET'] }
)

def get_channel_ids_from_crawl(crawl):
    query_info = client.start_query_execution(
        QueryString = f'''
            SELECT DISTINCT regexp_replace(
                url_path,
                '(\/channel\/[^\/]*)|(\/@[^\/]*)',
                '$1'
            )
            FROM "ccindex"."ccindex"
            WHERE crawl = '{crawl}'
            -- AND subset = 'warc'
            AND url_host_registered_domain = 'youtube.com'
            AND (
            url_path LIKE '/channel/%'
            OR url_path LIKE '/\@%'
            )
        ''',
        QueryExecutionContext = {
            'Database': 'ccindex'
        },
        ResultConfiguration = { 'OutputLocation': os.environ['AWS_S3_BUCKET'] }
    )

    query_id = query_info['QueryExecutionId']

    channel_ids = set()

    for _ in range(MAX_TIMEOUT_S):
        # wait for query to finish
        try:
            result = client.get_query_results(
                QueryExecutionId=query_id,
                MaxResults=1_000,
            )
            if result is not None:
                total = 0

                # iterate through all query results pages
                while 'NextToken' in result:
                    channel_ids |= set(row['Data'][0]['VarCharValue'].strip("/") for row in result['ResultSet']['Rows'])
                    total += len(result['ResultSet']['Rows'])

                    result = client.get_query_results(
                        QueryExecutionId=query_id,
                        NextToken=result['NextToken'],
                        MaxResults=1_000,
                    )

                channel_ids |= set(row['Data'][0]['VarCharValue'] for row in result['ResultSet']['Rows'])
                total += len(result['ResultSet']['Rows'])

                print(f"This Run: {total} | Unique: {len(channel_ids)}")
                return channel_ids

        except client.exceptions.InvalidRequestException as e:
            time.sleep(1)

def get_video_ids_from_crawl(crawl):
    query_info = client.start_query_execution(
        QueryString = f'''
            SELECT DISTINCT regexp_replace(
                url_path,
                '(?:\/embed\/(\S*))|(?:\/watch(\S*))|(?:\/v\/(\S*))',
                '$1'
            )
            FROM "ccindex"."ccindex"
            WHERE crawl = '{crawl}'
            --   AND subset = 'warc'
            AND url_host_registered_domain = 'youtube.com'
            AND (
                url_path LIKE '/watch%'
                OR url_path LIKE '/v/%'
                OR url_path LIKE '/embed/%'
            )
        ''',
        QueryExecutionContext = {
            'Database': 'ccindex'
        },
        ResultConfiguration = { 'OutputLocation': os.environ['AWS_S3_BUCKET'] }
    )

    query_id = query_info['QueryExecutionId']

    video_ids = set()

    for _ in range(MAX_TIMEOUT_S):
        # wait for query to finish
        try:
            result = client.get_query_results(
                QueryExecutionId=query_id,
                MaxResults=1_000,
            )
            if result is not None:
                total = 0

                # iterate through all query results pages
                while 'NextToken' in result:
                    video_ids |= set(row['Data'][0]['VarCharValue'].strip("/") for row in result['ResultSet']['Rows'])
                    total += len(result['ResultSet']['Rows'])

                    result = client.get_query_results(
                        QueryExecutionId=query_id,
                        NextToken=result['NextToken'],
                        MaxResults=1_000,
                    )

                video_ids |= set(row['Data'][0]['VarCharValue'] for row in result['ResultSet']['Rows'])
                total += len(result['ResultSet']['Rows'])

                print(f"This Run: {total} | Unique: {len(video_ids)}")
                return video_ids

        except client.exceptions.InvalidRequestException as e:
            time.sleep(1)

def crawl_videos():
    video_ids = set()

    for crawl in ALL_CRAWLS:
        prev_length = len(video_ids)
        video_ids |= get_video_ids_from_crawl(crawl)
        print(f"Total: {len(video_ids)} | Added: {len(video_ids) - prev_length}")

    # write video ids to JSON
    videos_list = list(video_ids)
    with open(YOUTUBE_VIDEOS_OUTPUT_FILE, "w") as f:
        json.dump(videos_list, f)

def crawl_channels():
    channel_ids = set()

    for crawl in ALL_CRAWLS:
        prev_length = len(channel_ids)
        channel_ids |= get_channel_ids_from_crawl(crawl)
        print(f"Total: {len(channel_ids)} | Added: {len(channel_ids) - prev_length}")

    # write channel ids to JSON
    channels_list = list(channel_ids)
    with open(YOUTUBE_CHANNELS_OUTPUT_FILE, "w") as f:
        json.dump(channels_list, f)

if __name__ == "__main__":
    # crawl_videos()
    crawl_channels()
            