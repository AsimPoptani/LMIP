import logging,os,datetime


from pull_data import DownloadHelper


def downloadDatasets(logger, unprocessed_data_dir="./data/unprocessed",tmp_dir="./tmp"):
    logger.info(f"Downloading datasets {datetime.datetime.now()}")
    # Get real path to data dir
    unprocessed_data_dir = os.path.abspath(unprocessed_data_dir)
    # Get real path to tmp dir
    tmp_dir = os.path.abspath(tmp_dir)
    # Append with file
    unprocessed_data_dir=os.path.join(os.path.dirname(__file__),unprocessed_data_dir)
    tmp_dir=os.path.join(os.path.dirname(__file__),tmp_dir)


    logger.info("Unprocessed data dir: {}".format(unprocessed_data_dir))
    logger.info("Tmp dir: {}".format(tmp_dir))
    # Create a download helper
    downloader = DownloadHelper(logger)



    downloader.download_zip_and_unzip("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip","maestro",data_dir=unprocessed_data_dir,tmp_dir=tmp_dir)


if __name__ == "__main__":
    # Set up a new logging with log name filter_midi.log 
    logging.basicConfig(filename="running.log",level=logging.INFO)

    # Create logger
    logger=logging.getLogger(name="DownloadDatasets")

    downloadDatasets(logger)
