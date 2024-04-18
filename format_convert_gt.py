import os

def process_files_in_folder(input_folder, output_folder):
    def read_data(data):
        '''
            Read the data from the txt file
        '''
        # Split the values from the data line
        class_name, x_center, y_center, width, height = data.split()
        
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)
        
        # Perform other processing operations if necessary
        
        # Return the processed values
        return class_name, x_center, y_center, width, height
        # Get the list of files in the input folder
    
    def process_data(class_name, x_center, y_center, width, height):
        ''' 
            input: x_center, y_center, width, height of bbox
                    ex: 0.913333     0.332787     0.033333       0.039344
            output: left, top, right, bottom of bbox
        '''
        
        # Calculate the coordinates of the bounding box
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2
        
        # Return the coordinates as a formatted string
        # Limit each value to 6 decimal places
        return f"{class_name} {left:.6f} {top:.6f} {right:.6f} {bottom:.6f}"
        
        
        
    files = os.listdir(input_folder)
    
    # Iterate through each file in the list
    for file_name in files:
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            
            with open(input_file_path, 'r') as input_file:
                lines = input_file.readlines()
                
                # Process each line in the file
                processed_lines = []
                for line in lines:
                    print(line)
                    class_name, x_center, y_center, width, height = read_data(line)
                    
                    # process data
                    
                    # Write the processed line to the list
                    processed_line = process_data(class_name, x_center, y_center, width, height)
                    processed_lines.append(processed_line)
                
                                # Write the processed lines to the output file
                with open(output_file_path, 'w') as output_file:
                    for line in processed_lines:
                        output_file.write(line + '\n')

input_folder = "...\\val"

output_folder = "...\\val_formatXYXY"

process_files_in_folder(input_folder, output_folder)