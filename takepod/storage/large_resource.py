import os
BASE_RESOURCE_DIR = "."

class LargeResouce:
    RESOURCE_NAME = "resource"
    URL = "url"
    ARHIVE = "archive"
    def __init__(self, **kwargs):
        self._check_args(kwargs)
        self.config = kwargs
        self.resource_location = os.path.join(
            BASE_RESOURCE_DIR,
            self.config[LargeResouce.RESOURCE_NAME])
        self._check_files()

    def _check_files(self):
        if os.path.exists(self.resource_location):
            return
        if self.config[LargeResouce.ARHIVE]:
            self._download_unarchive()
            return
        self._download(download_dir=self.resource_location)

    def _download(self, download_dir):
        pass

    def _unarchive(self):
        pass

    def _download_unarchive(self):
        pass

    def _check_args(self, arguments):
        essential_arguments = [LargeResouce.RESOURCE_NAME, LargeResouce.URL]
        for arg in essential_arguments:
            if arg not in arguments or not arguments[arg]:
                raise ValueError(arg+"must be defined"
                                 "while defining Large Resource")
