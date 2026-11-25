import json
import csv
import re
import os

# This script has only function definitions (python)

def extract_messages_from_json(input_file, output_txt_file):
    """
    Reads a JSON file, extracts the 'message' field from every message
    object, and saves them into a text file (one message per line).
    Specifically for the twitch-download pipeline setup.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract only the 'message' strings
        messages = [
            msg['message']
            for msg in data.get('messages', [])
            if 'message' in msg
        ]

        with open(output_txt_file, 'w', encoding='utf-8') as f:
            for msg in messages:
                f.write(msg + '\n')

        print(
            f"Successfully extracted {len(messages)} "
            f"messages to {output_txt_file}"
        )

    except Exception as e:
        print(f"Error extracting messages: {e}")


def transform_messages_to_vectors(
    input_txt_file,
    lookup_csv_file,
    output_json_file="default.json",
    return_dataset="false",
    print_unknown_words=False,
    valid_messages_file=None):
    """
    Takes a text file of messages and a CSV lookup table.

    The CSV should have two columns:
        word,integer

    Each message is transformed into a list of integers based on
    the lookup table.

    Example:
        "this is great"
        -> [5, 20, 100]

    The final JSON contains one vector per message.
    """
    
    lookup_table = {}
    dataset = []
    valid_messages = []

    # ------------------------------------------
    # 1. Load the lookup table
    # ------------------------------------------
    try:
        with open(lookup_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) != 2:
                    continue

                word, val = row

                lookup_table[word.lower().strip()] = int(val)

    except Exception as e:
        print(f"Error reading lookup table: {e}")
        return

    # ------------------------------------------
    # 2. Transform messages
    # ------------------------------------------
    dataset = []

    try:
        with open(input_txt_file, 'r', encoding='utf-8') as f:
        
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                # Convert to lowercase and split into words.
                #
                # Example:
                # "This is Great!"
                # -> ["this", "is", "great"]
                words = re.findall(r'\w+', line.lower())

               # Identify every unrecognized word
                unknown_words = [word for word in words
                                 if word not in lookup_table]

                if print_unknown_words and unknown_words:
                    print(f"Message {line_number}: " 
                          + ", ".join(unknown_words))


                # Convert each word to its integer ID.
                # Words not found in the lookup table are turned to 0.
                vector = [lookup_table.get(word, 0)
                          for word in words]


                # Only keep messages where at least one recognized word
                # was found in the lookup table.
                if any(vector):
                    dataset.append(vector)
                    valid_messages.append(line)




        # ------------------------------------------
        # 3. Save dataset
        # ------------------------------------------
        if valid_messages_file is not None:
            with open(valid_messages_file, 'w', encoding='utf-8') as f:
                json.dump(valid_messages, f, ensure_ascii=False)

        if return_dataset == "true":
            return dataset
        
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f)

        print(
            f"Successfully transformed {len(dataset)} "
            f"messages into vectors."
        )
        print(f"Saved dataset to {output_json_file}")

    except Exception as e:
        print(f"Error transforming messages: {e}")





def create_streamer_vectors(
    streamer_name,
    input_directory,
    lookup_csv_file,
    output_directory
):
    os.makedirs(output_directory, exist_ok=True)

    streamer_directory = os.path.join(
        input_directory,
        streamer_name
    )

    # Each original recording gets its own output folder.
    streamer_output_directory = os.path.join(
        output_directory,
        streamer_name
    )
    os.makedirs(streamer_output_directory, exist_ok=True)

    json_files = sorted([
        file for file in os.listdir(streamer_directory)
        if file.lower().endswith(".json")
    ])

    print(
        f"Found {len(json_files)} recordings for {streamer_name}"
    )

    extracted_file = os.path.join(
        output_directory,
        f"_temp_{streamer_name}.txt"
    )

    for i, json_file in enumerate(json_files, start=1):

        input_json = os.path.join(
            streamer_directory,
            json_file
        )

        # Preserve the original recording's filename.
        session_name = os.path.splitext(json_file)[0]

        output_file = os.path.join(
            streamer_output_directory,
            f"{session_name}.json"
        )

        print(
            f"[{i}/{len(json_files)}] Processing {json_file}"
        )

        # JSON -> messages
        extract_messages_from_json(
            input_json,
            extracted_file
        )

        # Messages -> vectors
        transform_messages_to_vectors(
            extracted_file,
            lookup_csv_file,
            output_file
        )

    if os.path.exists(extracted_file):
        os.remove(extracted_file)

    print(f"Finished {streamer_name}")
    print(f"Saved recordings to {streamer_output_directory}")

