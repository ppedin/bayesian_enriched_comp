# bayesian_enriched_comp
This folder contains code for a project developed in the context of my Ph.D. in Computational Linguistics at the University of Pisa (under the supervision of prof. Alessandro Lenci).

The goal of this project is to develop an instance of Neural RSA Agent (Andreas and Klein, 2016) that given a meaning to express, choose the most appropriate utterance to express it. In simple words, this agent is a sort of language models (in that it produces an utterances). 
Unlike traditional language models, RSA agents does not generate language solely based on probabilities learned during training, but they use these probabilities as priors for Bayesian inference. Here, Bayesian inference is made according to the RSA framework (Frank and Goodman 2012). The RSA framework ensure that inference is made based on pragmatic principles, in particular Grice's maxims. The goal is to make sure that the model produces language not only based on correspondences implicitly learned during training, but based on theoretical principles encoded in the inference formula.

Here we applied this framework with one novelty: instead of using images to represent meaning, we use logical formulas (according to formal theories of meaning such as formal semantics), that we encode using special meaning symbols. We use this framework to model the production of logical metonymies (sentences such as "I started the book", where we mean a covert event "reading" or "writing"). To experiment, we take inspiration from the paper by Fang et al. 2022: We build a seq2seq neural network that learns a semantics by mapping sequences of words to sequences of symbols. We then use this semantics as priors to guide the inference of a RSA agent. We show that using this framework the agent naturally learns to produce metonymies, without seeing them in the training set. 

If you are interested, please contact the author at pedinotti.paolo@gmail.com or ask permission to read the thesis in https://etd.adm.unipi.it/t/etd-06252024-203602/

Bibliography:
- Jacob Andreas and Dan Klein. 2016. Reasoning about Pragmatics with Neural Listeners and Speakers. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1173–1182, Austin, Texas. Association for Computational Linguistics.
- Frank, Michael C. and Noah D. Goodman. “Predicting Pragmatic Reasoning in Language Games.” Science 336 (2012): 998 - 998.
- Fang et al. 2022, Color Overmodification Emerges from Data-Driven Learning and Pragmatic Reasoning (https://www.researchgate.net/publication/360724573_Color_Overmodification_Emerges_from_Data-Driven_Learning_and_Pragmatic_Reasoning)

