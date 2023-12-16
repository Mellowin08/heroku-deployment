import unittest
from satisfaction_analysis import predict_sentiment

class TestPredictSentiment(unittest.TestCase):

    def test(self):
        test_data = [
            ("Positive", "This product is great! I love it! 😍. The texture is soft and it feels very comfortable in my skin 🤗"),
            ("Negative", "I ordered 4 of this product, but only 3 arrived 🫥😢! I am so disappointed with this!😫"),
            ("Negative", "They didn't even bother packaging it properly. It arrived and there are spills everywhere! It leaked during shipment (49% percent of the product is left). There are no safety labels!! >:( "),
            ("Positive", "This is 100% original. Received the product within 3 days! I really like the smell after using this. I only got this for 4$ which makes it affordable!"),
            ("Neutral", "IT WAS MEH. NOT THAT GOOD BUT NOT THAT BAD, THE TASTE COULD BE IMPROVED BUT IT'S ACTUALLY DOABLE."),
            ("Positive", "OH MY GOD!, THIS PRODUCT WORKS AS WONDERS. IT DIDN'T DISAPPOINT ME I WILL BUY THIS AGAIN!"),
            ("Negative", "I HAD A BAD EXPERIENCE USING THIS BEAUTY BAR. MY SKIN EXPERIENCED RASHES AND THE ITCHING WOUDLN'T STOP. I AM SO DISAPPOINTED AT THIS!."),
            ("Negative", "DON'T WASTE YOUR MONEY!!! This product is a piece of garbage. I have to hold it together with wire twisted around the parts. 1 star for the seller because the seller posted VERY deceitful pictures. This piece of garbage is VERY SMALL, fragile, and weak."),
            ("Positive", "Can fit my 14\" laptop. Stitching is neat, material is not so thick but not very thin either. Can be a good day-to-day tote, good quality for its price really and locally made!!")
        ]
        for expected_sentiment, review in test_data:
            with self.subTest(review=review):
                self.assertEqual(predict_sentiment(review)[0], expected_sentiment)

if __name__ == '__main__':
    unittest.main()
