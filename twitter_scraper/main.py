"""
CITS4404 Group C1
Scrapes Twitter and writes tweets to file
"""
import os, sys, re, yaml, json
import tweepy
import time

from datetime import datetime

from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener

class QueueListener(StreamListener):
    def __init__(self):
        """ Creates a new stream listener with an internal queue for tweets """
        super(QueueListener, self).__init__()
        self.num_handled = 0
        self.queue = []
        self.batch_size = 20

        # twitter config
        cfg = yaml.load(open('config.yml', 'rt'))['twitter']
        self.auth = OAuthHandler(cfg['consumer_key'], cfg['consumer_secret'])
        self.auth.set_access_token(cfg['access_token'], cfg['access_token_secret'])
        self.api = tweepy.API(self.auth)

        if not os.path.exists('twitter_data'):
            os.makedirs('twitter_data')
        self.dumpfile = "twitter_data/%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S")

    def on_data(self, data):
        """ Routes the raw stream data to the appropriate method """
        raw = json.loads(data)
        if 'in_reply_to_status_id' in raw:
            if self.on_status(raw) is False:
                return False
        elif 'limit' in raw:
            if self.on_limit(raw['limit']['track']) is False:
                return False
        return True

    def on_status(self, raw):
        if isinstance(raw.get('in_reply_to_status_id'), int):
            # print("(%s)%s / %i" % (raw['in_reply_to_status_id'], raw['text'], len(self.queue)))
            line = (raw.get('in_reply_to_status_id'), raw.get("text"))
            self.queue.append(line)
            if len(self.queue) >= self.batch_size: self.dump()
        return True

    def on_error(self, status):
        print('On Error:', status)

    def on_limit(self, track):
        print('On Limit:', track)

    def dump(self):
        pcnt = 0
        with open(self.dumpfile, 'a') as fdump:
            (sids, texts), self.queue = zip(*self.queue), []
            while True:
                try:
                    lines_mapper = {s.id_str: s.text for s in self.api.statuses_lookup(sids)}
                    break
                except Exception as e:
                    print("Error", e)
                    time.sleep(10)
            lines_grps = [[lines_mapper.get(str(sid)), txt] for sid, txt in zip(sids, texts) if
                          lines_mapper.get(str(sid))]
            lines_grps = [[self.preprocess(s) for s in lines] for lines in lines_grps]

            for lines in lines_grps:
                for i in range(len(lines) - 1):
                    if self.is_ascii(lines[i]) and self.is_ascii(lines[i + 1]):
                        fdump.write("%s\n%s\n" % (lines[i], lines[i + 1]))
                        pcnt += 1
        self.num_handled += pcnt

    def preprocess(self, line):
        line = re.sub("\s+", ' ', line).strip().lower()
        return line

    def is_ascii(self, line):
        try:
            encoded_line = line.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False


def main():
    listener = QueueListener()
    stream = Stream(listener.auth, listener)

    stream.filter(locations=[-122.75, 36.8, -121.75, 37.8, -74, 40, -73, 41,
                             150, -34, 151, -33, 144.65, -38, 145.4, -37.7,
                             115.7, -32.5, 116.02, -31.7], languages=['en'])
    # stream.filter(languages=["en"], track=['python', 'obama', 'trump'])

    try:
        while True:
            try:
                stream.sample()
            except KeyboardInterrupt:
                print('Keyboard Interrupted')
                return
    finally:
        stream.disconnect()
        print('Exit successful, twitter_data dumped in %s' % listener.dumpfile)


if __name__ == '__main__':
    sys.exit(main())
