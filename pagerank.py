import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    trans_mod = {}

    #number of files in corpus
    num_files = len(corpus)

    # get number of links to page
    num_links = len(corpus[page])

    if num_links != 0:
        # calculate random probability (which is application for all pages)
        rand_prob = (1 - damping_factor) / num_files
        # calculate specific page-related probability
        spec_prob = damping_factor / num_links
    else:
        #calculate random probability(which is applicable for all pages)
        rand_prob = (1 - damping_factor) / num_files
        #calculate specific page-related probability
        spec_prob = 0

    # iterate over files
    for file in corpus:
        #check if current page has any links
        if len(corpus[page]) == 0:
            trans_mod[file] = 1 / num_files
        else:
            # if file is not current page, there is no need to get its links
            if file not in corpus[page]:
                trans_mod[file] = rand_prob
            else:
                if file not in corpus[page]:
                    trans_mod[file] = rand_prob
                else:
                    trans_mod[file] = spec_prob + rand_prob
    # checks if sum of probabilities is 1
    if round(sum(trans_mod.values()),5) != 1:
        print(f'ERROR! probabilities add up to {sum(trans_mod.values()),}')
    #else:
    return trans_mod


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_PR = {}
    for page in corpus:
        sample_PR[page] = 0

    # simple is initially none
    sample = None 

    for iteration in range(n):
        #if sample is None (in first iteration)
        if sample ==  None:
            # list of all choices 
            choices = list(corpus.keys())
            # random choice a simple  --> random.choice(choices)
            sample = random.choice(choices)
            sample_PR[sample] += 1
        else:
            # get transition model for current sample
            next_sample_prob = transition_model(corpus, sample, damping_factor)    # come to this line if this doesnt work
            # get random choice
            choices = list(next_sample_prob.keys())
            weights  =  [next_sample_prob[key] for key in choices]
            sample = random.choices(choices, weights).pop()
            sample_PR[sample] += 1
    # after sampling is finished --> divide by number of iterations
    sample_PR = {key: value / n for key, value in sample_PR.items()}
    if round (sum(sample_PR.values()),5) != 1:
        print(f'ERROR! probabilities add up to {sum(transition_model.values()),}') # come back here later 
    else:
        print(f'ERROR! probabilities add up to {round(sum(sample_PR.values()),10)}')
    return sample_PR

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create empty dictionary
    iterate_PR = {}
    # safe the number of pages in variables 
    num_pages = len(corpus)
    # iterate over pages
    for page in corpus:
        iterate_PR[page] = 1/num_pages

    changes = 1
    iterations = 1
    while changes >= 0.001:
        changes = 0
        previous_state = iterate_PR.copy()
        for page in iterate_PR:
            parents = [link for link in corpus if page in corpus[link]]
            first = ((1 - damping_factor) / num_pages)
            second = []
            if len(parents) != 0:
                for parent in parents:
                    num_links = len(corpus[parent])
                    val = previous_state[parent] / num_links
                    second.append(val)

            second = sum(second)
            iterate_PR[page] = first + (damping_factor * second)
            new_change = abs(iterate_PR[page] - previous_state[page])
            if changes < new_change:
                changes = new_change
        iterations += 1
    dictsum = sum(iterate_PR.values())
    iterate_PR = {key: value / dictsum for key, value in iterate_PR.items()}
    print(f'\nPageRank value stable after {iterations} iterations.')
    print(f'sum of iterate_PR values: {round(sum(iterate_PR.values()), 10)}')
    return iterate_PR

if __name__ == "__main__":
    main()
