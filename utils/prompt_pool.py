class PromptGenerator:
    def prompt(self, num_classes, class_info, envision_nums=50):
        return f"""Q: I have gathered images of 4 distinct categories: ['Husky dog', 'Garfield cat', 'churches', 'truck']. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify 5 categories that visually resemble to these broad categories but have no direct relation to these broad categories. Please list these 5 items for me.
A: These 5 items are:
- black stone
- mountain
- Ginkgo Tree
- river
- Rapeseed

Q: I have gathered images of {num_classes} distinct categories: [{class_info}]. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify {envision_nums} classes that visually resemble to these broad categories but have no direct relation to these broad categories. Please list these {envision_nums} items for me.
A: These {envision_nums} items are:
"""

    def get_prompt(self, num_classes, class_info=None, envision_nums=50):
        return self.prompt(num_classes, class_info, envision_nums)

    def prompt_again(self, envision_nums=50):
        return f"""Q: Give me {envision_nums} more categories that are visually similar to these broad categories you summarized in the dataset but have no direct relation to these broad categories. Each category you give cannot exceed three words and could not have appeared in your previous answers.
A: The other {envision_nums} categories are:
"""

    def get_prompt_again(self, envision_nums=50):
        return self.prompt_again(envision_nums)
