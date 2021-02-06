import coinbasepro as cbp
import decimal
import json
from datetime import datetime
from datetime import timedelta


class BackFiller:
    def determine_end_date(self, start: str, end: datetime):
        """[summary]
            Given the start date and the final end date, will determine the end date with a 7 day step period.
        Args:
            start (str): [description] The start date
            end (datetime): [description] The final end date

        Returns:
            str: [description] Returns the iso formatted date string computed
        """
        date = datetime.fromisoformat(start)
        diff_delta = end - date
        if diff_delta.days < 7:
            delta = diff_delta
        else:
            delta = timedelta(days=7)
        return (date + delta).isoformat()

    def fill_candle_data(
        self,
        product_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity=3600,
    ):
        """[summary]
            Given the inputs, does batched calls to return historical candle dataset.
        Args:
            product_id (str): [description] The product to return data for (i.e BTC-USD)
            start_date (datetime): [description] The start date to query historical data for
            end_date (datetime): [description] The end date for the historical data period
            granularity (int, optional): [description]. Defaults to 3600. Time in seconds to pull candle data for

        Returns:
            array: [description] The flattened historical data for the product
        """
        candle_data = []
        client = cbp.PublicClient()
        start_month = start_date.month
        new_start_date = None
        end_date_delta = None
        while start_date <= end_date:
            if new_start_date is None:
                new_start_date = start_date.isoformat()
            else:
                new_start_date = end_date_delta
            end_date_delta = self.determine_end_date(new_start_date, end_date)
            candle_data.append(
                client.get_product_historic_rates(
                    product_id, new_start_date, end_date_delta, granularity
                )
            )
            start_date = datetime.fromisoformat(new_start_date)
        return [
            item for sublist in candle_data for item in sublist
        ]  # flatten the data

    class CandleDataEncoder(json.JSONEncoder):
        """[summary]
            Custom encoder to handle issues for writing datetime and decimal objects.
        Args:
            json (class): [description] The required default JSONEncoder
        """

        def default(self, o):
            if isinstance(o, datetime):
                return o.isoformat()
            if isinstance(o, decimal.Decimal):
                return float(o)
            return super(CandleDataEncoder, self).default(o)

    def backfill_data(
        self,
        product_id="BTC-USD",
        train_start_date="2020-05-28T01:00:00.000",
        train_end_date="2020-10-28T01:00:00.000",
        test_start_date="2020-10-28T01:00:00.000",
        test_end_date="2020-12-28T01:00:00.000",
        train_data_file="datasets/train_data.txt",
        test_data_file="datasets/test_data.txt",
    ):
        """[summary]
            Will backfill the historical data for the provided product with the given time periods
            and write to the given file locations.
        Args:
            product_id (str, optional): [description]. Defaults to 'BTC-USD'. The product to backfill for
            train_start_date (str, optional): [description]. Defaults to "2020-05-28T01:00:00.000". Training data start date
            train_end_date (str, optional): [description]. Defaults to "2020-10-28T01:00:00.000". Training data end date
            test_start_date (str, optional): [description]. Defaults to "2020-10-28T01:00:00.000". Test data start date
            test_end_date (str, optional): [description]. Defaults to "2020-12-28T01:00:00.000". Test data end date
            train_data_file (str, optional): [description]. Defaults to "datasets/train_data.txt". Location to save training data
            test_data_file (str, optional): [description]. Defaults to "datasets/test_data.txt". Location to save test data
        """
        train_data = self.fill_candle_data(
            product_id,
            datetime.fromisoformat(train_start_date),
            datetime.fromisoformat(train_end_date),
        )
        test_data = self.fill_candle_data(
            product_id,
            datetime.fromisoformat(train_end_date),
            datetime.fromisoformat(test_end_date),
        )
        with open(train_data_file, "w") as filehandle:
            json.dump(train_data, filehandle, cls=CandleDataEncoder)

        with open(test_data_file, "w") as filehandle:
            json.dump(test_data, filehandle, cls=CandleDataEncoder)

    def load_training_data(
        self,
        train_data_file="datasets/train_data.txt",
        test_data_file="datasets/test_data.txt",
    ):
        """[summary]
            Loads the training data into arrays based on given file locations.
        Args:
            train_data_file (str, optional): [description]. Defaults to "datasets/train_data.txt". Location to load training data
            test_data_file (str, optional): [description]. Defaults to "datasets/test_data.txt". Location to load test data

        Returns:
            [type]: [description]
        """
        with open(train_data_file, "r") as filehandle:
            train_data = json.load(filehandle)

        with open(test_data_file, "r") as filehandle:
            test_data = json.load(filehandle)
        return train_data, test_data
