# Drug Wizard: Your Pill Identifier and Companion

### Inspiration: 

The biggest inspiration for our project was to improve medication safety. 131 million or 66% of U.S. adults take prescription drugs and more than 530,000 injuries occur annually in outpatient clinics due to medication errors. Drug Wizard is designed to help individuals, clinicians, caretakers, and more to ensure that the correct medication is provided as well as at the right dosage. Reducing the risk of medication errors was also personal to us and our group. One of our members erroneously took the wrong medication and almost endured a serious injury because of it. Overall, we just wanted to have a positive healthcare impact for individuals and healthcare professionals alike.

### What it does: 

Our app allows the user to upload an image of their unidentified pill. The pill is identified based on color, shape and text. Those fields are then ran through the National Library of Medicine’s ‘pillbox’ data set to find any matches. The top matches will be shown to the user.

### How we built it:

We started off using Open CV to contour the pills based on color. Color ranges were set using Hue Saturation Value (HSV) color model, and then masks were created of the image. Using these masks, a contour could be created. By checking the contour area by color, we could determine which color the pill was. The contour also allowed us to determine the shape based on the amount of approximated edges. We then used pandas to manipulate the ‘pillbox’ data set to find any matches. We used streamlit framework to allow for file uploading for the user. We also integrated Google's cloud vision API in order to extract text from the pill engravings. Furthemore, we connected OpenFDA's API in order to get drug interactions based on what we determined the pill to be, and added a web scraper to garner relevant search results given the text interpreted off of the pill.

### Challenges we ran into:

Developing our app was both exciting and challenging, mainly because we encountered a lot of new technologies. Another constraint shared by many was time management. Because the Pitt Challenge was limited to only 42 hours, trying to create a fully functioning prototype by cobbling together a variety of Python modules was an experience, to say the least. Our string manipulation of data garnered from Google's vision API isn't perfect, although could be perfected with a little more time. Furthermore, we ran into issues with scheduled maintenance for key databases and API access when we really needed them which was frustrating. OpenFDA's API was also difficult to get started with, but we look to improve on this as well.

### Accomplishments that we're proud of:

Being able to create masks and contours of the pills was one of our first big milestones. It was the first moment when we were like “Oh... We can actually do this.“ We are still only able to use a localhost for Streamlit, but getting that working with little prior app development felt good. Also implementing Google's vision API was a sigh of relief after attempting OCRs and a neural network. All in all, it just feels good to come away with a completed project, as most of our team has never done a hackathon before.

### What we learned

Working through the project, we received a variety of valuable learning experiences and skills. Firstly, we learned about how to assemble a project. It was one of our first experiences with live coding in Github, and although it's messy, we came away unscathed. We also came away with a better understanding of pixel values in images to create the color hues. Open CV made it really easy to take those to a mask, and learning about computer vision was a new topic for all of us. We also learned about the implementation of APIs, and how simple and useful the calls can be.

### What’s next for Drug Wizard?

Currently, our main focus on how to move forward with Drug Wizard is to enhance accuracy, usability, and features. The project is still very young and could greatly benefit from some refinement. To refine the search, we need to add more color values and more shapes and have better string manipulation to search. The regex we used to find the imprint in the data frame sometimes was one letter, such as O, which would cause an influx of applicable pills. After refining the search itself, the next step would be market research to determine the demand for the app and what features would be best to implement.  We also want to get our OpenFDA API working to gather pill interactions to provide warnings to the user. An application to solely identify pills could be useful, but adding significant features such as gathering allergy information from a barcode scan of a food item could really improve the app’s potential commercialization. 

Built with:
Python3
OpenCV
Numpy
Pillow
Pytesseract
Matplotlib.pyplot
Pandas
Beautiful Soup
Requests
Jupyter
Streamlit
Google Cloud Vision API
TempFiles
Time