# A sample dataset of GenDD and PADD for food-ordering domain is given in the folder GenPADS_Data_Sample.

# The GenPADS_Data_Sample folder contains

```
|_______/Sample_GenDD
|	       |__/Cleaned Dataset
|	       	|__informatively_filtered_food-ordering.csv		# cleaned Taskmaster Dataset for food-ordering domain.
|
|	       |__/DG-Dataset
|	       	|___ formatted_food-ordering.csv				# User-Bot sequence-to-sequence dataset to train the Dialogue Generator Model for food-ordering domain.
|	       	|___ formatted_food-ordering_train.csv			# train split of seq2seq food-ordering domian DG-dataset.
|	       	|___ formatted_food-ordering_test.csv                 # test split of seq2seq food-ordering domian DG-dataset.
|                 |___ formatted_food-ordering_val.csv 			# val split of seq2seq food-ordering domian DG-dataset.
|
|	       |__/G-Dataset
|	       	|___ food-ordering_generator_Bot.csv			# Bot-Bot sequence-to-sequence dataset to train the Generation Module (G) for food-ordering domain.
|	       	|___ food-ordering_generator_bot_train.csv		# train split of seq2seq food-ordering domian G-dataset.
|	       	|___ food-ordering_generator_bot_test.csv             # test split of seq2seq food-ordering domian G-dataset.
|                 |___ food-ordering_generator_Bot_val.csv 			# val split of seq2seq food-ordering domian G-dataset.
|
|_______/Sample_PADD
|	       |__/flights_pol_annotated_data.csv					# a sample of politeness annotated flights domain dataset to train a politeness classifier (PC) in flight domain.
|	       |__/food-ordering_pol_annotated_data.csv				# a sample of politeness annotated food-ordering domain dataset to train a politeness classifier (PC) in food-ordering domain.
|	       |__/hotels_pol_annotated_data.csv					# a sample of politeness annotated hotels domain dataset to train a politeness classifier (PC) in hotels domain.
|	       |__/movies_pol_annotated_data.csv					# a sample of politeness annotated movies domain dataset to train a politeness classifier (PC) in movies domain.
|	       |__/music_pol_annotated_data.csv					# a sample of politeness annotated music domain dataset to train a politeness classifier (PC) in music domain.
|	       |__/restaurant-search_pol_annotated_data.csv			# a sample of politeness annotated restaurant-search domain dataset to train a politeness classifier (PC) in restaurant-search domain.
|	       |__/sports_pol_annotated_data.csv					# a sample of politeness annotated sports domain dataset to train a politeness classifier (PC) in sports domain.


> /Sample_GenDD/Cleaned Dataset/informatively_filtered_food-ordering.csv contains six columns: 
	> turn_id: denotes the turn number in an ongoing conversation.
	> speaker: gives the speaker role of uttered utterance i.e. USER/ASSISTANT.
	> text: denotes the uttered utterance text.
	> conversation_id: gives the unique conversation ID for each conversation.
	> segments_annotations_name: represents the slot-value information which is requested/informed.
	> segments_text: gives the extracted slot-value/information from the utterance.
	> utterance_id: denotes the unique utterance ID of an utterance in a conversation.

> /Sample_GenDD/DG-Dataset/formatted_food-ordering.csv contains two columns: 
	> User: denotes the user's uttered utterance in an ongoing dialogue.
	> Bot: denotes the bot's uttered utterance for a corresponding input user's utterance in an ongoing dialogue.

> /Sample_GenDD/G-Dataset/food-ordering_generator_Bot.csv contains two columns: 
	> input_text: denotes the bot's developer-designed template response for a selected action.
	> target_text: gives the different possible semantically similar but diverse responses for a given input_text. 

> /Sample_PADD/<domain_name>_pol_annotated_data.csv contains two columns:
	> text: denotes the utterance text.

	> labels: denotes one of the four polite classes: 0/1/2/4.
		>> 0: impolite
		>> 1: somewhat_impolite
		>> 2: somewhat_polite 
		>> 4: polite

