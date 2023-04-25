# Iris-Logistic-Regression
Iris Logistic Regression Without scikit learn
Implementation of logistic regression on Iris data


Today, there are several machine learning models. Different types of models, like different people, have their own specialties and areas in which they excel more than others. Here in this paper we are going to use logistic regression model services in iris data set to predict flower species. Logistic regression is used to predict a dependent variable, given a set of independent variables, such that the dependent variable is classified. The Logistic function is an S-shaped curve that can take any real-valued number and map it to a value between 0 and 1.

Lily of the valley
It contains three types of irises with 50 examples as well as some properties about each flower. One species of flower can be linearly separated from two other species, but the other two species cannot be separated from each other linearly.
 

The columns of this data set are:

ID = ID
Sepal length cm = SepalLengthCm
Sepal Width cm = SepalWidthCm
Petal length cm = PetalLengthCm
Petal Width cm = PetalWidthCm
Species = Species

The purpose of this work is to identify different types of iris based on the length and width of the sepals and the length and width of their petals.
For this, we first import the desired libraries.
Then we import the data set and display them to ensure the validity of the data.
With the describe command, we return a description about the data.
If the DataFrame contains numeric data, the description contains this information for each column:

count - number of non-empty values.
Average - average value (average).
std - standard deviation.
min - minimum value.
25% - 25% percent*.
50% - 50% percent*.
75% - 75% percentile*.
max - the maximum value.

We get more details with the info command.
It is used with the corr() command to find the pairwise correlation of all in Python.
With the Species.unique command, we remove duplicate data to determine the category of lilies.
To analyze and better understand the difference between species, we draw them.
To check the difference between species, we draw them.
We draw the species based on length and width and all of them on the graph to get the dispersion of the data.
In the pre-processing stage, we delete and edit the data that are not useful to us.

Logistic regression step
In the first part of this step, we perform normalization so that the data close to each other become the same.
In the second part, we separate the test and training data.
In the third part, we create and initialize the parameters.
In the fourth part, we create the sigmoid function.
In the fifth section, we create forward and backward functions for training data.
In the sixth section, we created a function to update weights and bias.
In the seventh section, we created the prediction function.
In the eighth step, we create a function for logistic regression.
Then we run the logistic regression function.
In the output, for every 50 iterations, we will see the error rate, and then we will see the standard deviation chart, and at the end, we will see the correctness of the code.

# ===================> Fa  v

پیاده سازی رگرسیون لاجستیک بر روی داده Iris


امروزه چندین مدل یادگیری ماشین وجود دارد. انواع مختلف مدل ها، مانند افراد مختلف، تخصص ها و زمینه های خود را دارند که در آن بیشتر از دیگران برتری دارند. در اینجا در این مقاله قصد داریم از خدمات مدل رگرسیون لجستیک در مجموعه داده زنبق برای پیش‌بینی گونه‌های گل استفاده کنیم. رگرسیون لجستیک برای پیش‌بینی یک متغیر وابسته، با توجه به مجموعه‌ای از متغیرهای مستقل، به‌گونه‌ای که متغیر وابسته طبقه‌بندی می‌شود، استفاده می‌شود. تابع Logistic یک منحنی S شکل است که می تواند هر عدد با ارزش واقعی را گرفته و آن را به مقداری بین 0 و 1 ترسیم کند.

گونه زنبق
این شامل سه گونه زنبق با 50 نمونه و همچنین برخی از خواص در مورد هر گل است. یک گونه گل به صورت خطی از دو گونه دیگر قابل تفکیک است، اما دو گونه دیگر به صورت خطی از یکدیگر قابل جدا شدن نیستند.
 

ستون های این مجموعه داده عبارتند از:

شناسه  = ID
طول کاسبرگ سانتی متر = SepalLengthCm
پهنای کاسبرگ سانتی متر = SepalWidthCm
طول گلبرگ سانتی متر = PetalLengthCm
پهنای گلبرگ سانتی متر = PetalWidthCm
گونه ها = Species

هدف از این کار شناسایی گونه های مختلف زنبق بر اساس طول و پهنای کاسبرگ و طول و پهنای گلبرگ آنها است.
برای اینکار ابتدا کتابخانه های مورد نظر را وارد میکنیم.
سپس دیتا ست را وارد کرده و برای اطمینان از معتبر بودن داده ها، آنها را نمایش میدهیم.
با دستور describe توضیحاتی را در مورد داده ها بر میگردانیم.
اگر DataFrame حاوی داده های عددی باشد، توضیحات حاوی این اطلاعات برای هر ستون است:

count - تعداد مقادیر غیر خالی.
میانگین - مقدار متوسط (میانگین).
std - انحراف استاندارد.
min - حداقل مقدار.
25% - صدک 25%*.
50% - صدک 50%*.
75% - صدک 75%*.
max - حداکثر مقدار.

با دستور info به جزئیات بیشتری میرسیم.
با دستور corr() برای یافتن همبستگی زوجی همه در پایتون استفاده میشود.
با دستور Species.unique داده های تکراری را حذف میکنیم تا دسته بندی زنبق ها مشخص بشود.
برای آنالیز و درک بهتر تفاوت گونه ها آنها را رسم می کنیم.
برای بررسی تفاوت بین گونه ها آنها را رسم می کنیم.
گونه ها را بر اساس طو و عرض و همه آنها را روی نمودار رسم میکنیم تا پراکندگی داده ها را بدست آوریم.
در مرحله پیش پردازش، داده هایی که به کارمون نمیان رو حذف و ویرایش میکنیم.

مرحله رگرسیون لاجستیک
در بخش اول این مرحله، نرمال سازی رو انجام میدهیم تا داده های نزدیک به هم یکی شوند.
در بخش دوم داده های تست و ترین را جدا می کنیم.
در بخش سوم، پارامترها را ایجاد و مقدار دهی اولیه می کنیم.
در بخش چهارم، تابع سیگموئید را ایجاد می کنیم.
در بخش پنجم، تابع پیش رونده و پس رونده را برای آموزش داده ها ایجاد می کنیم.
در بخش ششم، تابعی برای بروزرسانی وزن ها و بایاس ایجاد کردیم.
در بخش هفتم، تابع پیش بینی را ایجاد کردیم.
در مرحله هشتم، تابعی برای رگرسیون لاجستیک ایجاد میکنیم.
سپس تابع رگرسیون لاجستیک را اجرا میکنیم.
در خروجی به ازای هر 50 تا ایتریشن میزان اشتباه را مشاهده میکنیم و سپس نمدار انحراف معیار را خواهیم دید و  در انتها میزان درستی کد را مشاهده مینماییم.



END
