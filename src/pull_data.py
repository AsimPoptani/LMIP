import requests , zipfile, os, logging
from tqdm import tqdm


class DownloadHelper:
    def __init__(self,logger:logging.Logger=logging.getLogger(name="DownloadHelper")) -> None:
        self._logger:logging.Logger = logger

    # Pull a zip file from the given URL
    def pull_zip(self,url:str,data_dir:str, zip_file:str) -> None:
        # Log that we are pulling the zip file
        self._logger.info("Pulling zip file: {}".format(url))
        # Path
        path = os.path.join(data_dir,zip_file)
        # Check if the zip file exists
        if os.path.exists(path):
            self._logger.info("Zip file already exists: {}. Skipping.".format(zip_file))
            # Quit if the zip file exists
            return

        # Pull the zip file
        r = requests.get(url, stream=True)
        # Print current path
        print(os.getcwd())
        # Write the zip file
        with open(path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
                    f.flush()

    def unzip(self,zip_file:str, folder:str) -> bool:
        # Check if the zip exists
        if not os.path.exists(zip_file):
            self._logger.error("Zip file does not exist: {}.".format(zip_file))
            # Return false
            return False
        
        # Check if the folder exists
        if not os.path.exists(folder):
            # Create the folder
            self._logger.info("Folder does not exist: {}. Creating.".format(folder))
            os.makedirs(folder)

        # Unzip the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            self._logger.info("Unzipping: {}".format(zip_file))
            zip_ref.extractall(folder)

    def download_data(self,url:str,folder:str,name:str) -> None:
        # Download the file
        self._logger.info("Downloading: {}".format(url))

        # Create the folder if it doesn't exist and not a place downloaded data there
        if not os.path.exists(folder):
            self._logger.info("Folder does not exist: {}. Creating.".format(folder))
            os.makedirs(folder)

        
        # Download the file
        r=requests.get(url, stream=True)
        # Write the file
        with open(os.path.join(folder,name), 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
                    f.flush()
    
    def delete_file_folder(self,file_or_folder:str) -> bool:
        # Check if the file or folder exists
        if os.path.exists(file_or_folder):
            # Delete the file or folder
            self._logger.info("Deleting: {}".format(file_or_folder))
            os.remove(file_or_folder)
            return True
        
        else:
            self._logger.info("File or folder does not exist: {}. Skipping.".format(file_or_folder))
            return False
        
    def download_zip_and_unzip(self,url:str,folder:str,data_dir:str="../data", tmp_dir:str="../tmp",cache:bool=True,skip:bool=True) -> None:
        """Downloads and extracts a zip file.

        Args:
            url (str): URL of the zip file.
            folder (str): Name of the folder to extract the zip file to.
            data_dir (str, optional): Place where the data will be extracted. Defaults to "../data".
            tmp_dir (str, optional): A temporary cache place where data can be put. Defaults to "../tmp".
            cache (bool, optional): Whether to delete data to save space. Defaults to True.
            skip (bool, optional): Checks if the dataset exists and if it does then don't download it again. Defaults to True.
        """
        if skip:
            # Check if the folder exists
            if os.path.exists(os.path.join(data_dir,folder)):
                self._logger.warning("Dataset exists: {}. Skipping.".format(folder))
                # Quit if the folder exists
                return

        # zip file name
        zip_file= url.split('/')[-1]
        # Download the file
        self.pull_zip(url,tmp_dir,zip_file)

        print(os.path.join(tmp_dir,folder),os.path.join(data_dir,folder))
        # Unzip the file
        self.unzip(os.path.join(tmp_dir,zip_file),os.path.join(data_dir,folder))
        if not cache:
            # Delete the zip file
            self.delete_file_folder(os.path.join(tmp_dir,zip_file))
     