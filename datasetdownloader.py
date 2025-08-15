import SoccerNet

from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=r"dataset")
mySoccerNetDownloader.password = "s0cc3rn3t"

# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])
# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
# mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train","test","challenge"]) #? Downloaded
# mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"], password="PW_FROM_NDA")
print("Execution completed")
