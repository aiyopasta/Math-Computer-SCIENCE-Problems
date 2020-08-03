# LOOK THROUGH ENTIRE CODE AND DELETE UNNECESSARY COMMENTS!

import numpy as np
import matplotlib.pyplot as plt

# Just to group together a bunch of repeated variables.
class Server:
    def __init__(self, lambda_):
        # Lambda parameter for exponential distribution from which we sample service times.
        # Lambda here is the MEAN.
        self.lamb = lambda_

        # To answer question 2
        self.total_service_time = 0
        self.n_customers_served = 0

        # Break times (Times when this server takes its 2 breaks).
        # FOR NOW, IT IS 500 for both breaks as we don't yet know what to do if there is a scheduling conflict.
        self.break_times = [np.random.random() * 2*60, (np.random.random() * 2*60) + (2*60)]
        self.n_breaks_left = 2

        # The time when server becomes free (either from a break or from serving someone).
        # When the current time is greater than or equal to this time, the server is considered free.
        self.t_when_free = 0

    def next_break_time(self):
        if self.n_breaks_left == 0:
            return (4*60) + 1000  # never

        elif self.n_breaks_left == 1:
            return self.break_times[1]

        else:  # still has both breaks available
            return self.break_times[0]

    # Call this when there is a conflict between a scheduled service time and a break.
    def shift_breaks(self, new_start_time):
        assert self.n_breaks_left > 0

        if self.n_breaks_left == 2:
            self.break_times[0] = new_start_time

            # There is the possibility that we have to shift the other break down as well, if shifting the first
            # break causes it to overlap with the second one.
            if self.break_times[1] < self.break_times[0]+5:
                self.break_times[1] = self.break_times[0] + 5

        else:  # if 1 break left
            self.break_times[1] = new_start_time

    # Reset Breaktime Variables and t_when_free variable (call when new epoch starts)
    def reset(self):
        self.break_times = [np.random.random() * 2 * 60, (np.random.random() * 2 * 60) + (2 * 60)]
        self.n_breaks_left = 2
        self.t_when_free = 0



# MAIN FUNCTION
def run():
    ''' Creation of all the variables'''

    n_epochs = 100

    ''' Main simulation (ALSO REMEMBER TO ADD A FOR-LOOP THAT RUNS THIS 100 TIMES PER k) '''
    for k in {3, 4, 5}:
        # To answer question 1:
        total_interarrival_time = 0
        total_customers_entered = 0

        # To answer question 2 (see inside "Server" class):
        s1, s2 = Server(1.2), Server(1.5)

        # NOTE: To answer question 3, we just have to divide the total_service_time (for each server) by 4*60,
        # the total time.

        # To answer questions 4 + 5:
        n_total_waiters = 0
        total_waiting_time = 0

        # To answer question 6:
        n_custs_who_left = 0

        # Other vars
        t = 0  # current_time
        t_next_cust = 0  # time when next customer will arrive
        cust_entered_times = []  # e.g. [10, 4, 5] means the customer at front of line entered at t=10, the customer
        # second in line entered at t=4, and the third customer at the back of the line is
        # at t=5.

        # Histogram vars –––––––––
        all_customer_arrival_times = []  # for histogram 1
        all_interarrival_times = []      # for histogram 2
        all_server1_service_times = []   # for histogram 3
        all_server2_service_times = []   # also for histogram 3
        # ––––––––––––––––––––––––

        for epoch in range(n_epochs):
            # Reset / Initialize some variables
            t_next_cust = exp(k)
            s1.reset()
            s2.reset()
            cust_entered_times = []

            all_customer_arrival_times.append(t_next_cust)
            all_interarrival_times.append(t_next_cust)

            # Draw initial interarrival sample
            total_interarrival_time += t_next_cust
            t = t_next_cust
            while t < 4*60:
                print('t='+str(t))

                ''' Handle a new incoming customer. Either they enter line, or leave. (Note that they could enter the line
                    and immediately leave in the following processing step, but this won't count as them having waited a turn.)'''
                if t == t_next_cust:
                    print('We came to this time step because there is a new customer here.')

                    # Update total customer count
                    total_customers_entered += 1

                    # Add customer's enter time to waiting line (but if there are 4 people already there, they might leave
                    # with probability 50% and if there are 5 or more already there, they might leave with probability 60%).
                    # NOTE: Newly added customer to line might be instantly removed in the processing step (below)
                    # if they are the only waiter).
                    n = len(cust_entered_times)
                    if n <= 3:
                        cust_entered_times.append(t)
                        print('There are only', n, ' people in line so the customer was added to the line.')

                    elif n == 4:
                        print('There are exactly 4 people already in line, so the new customer might leave.')
                        if np.random.random() <= 0.5:
                            cust_entered_times.append(t)
                            print('Aye!! They stayed. Now there are', len(cust_entered_times), ' people in line.')
                        else:
                            print('Rip lol they left. There are still', n, ' people in line.')
                            n_custs_who_left += 1

                    else:  # if n >= 5
                        print('There are <= 5 people already in line, so the new customer might leave.')
                        if np.random.random() <= 0.4:  # they stay with probability 40% (leave with 60% prob.)
                            cust_entered_times.append(t)
                            print('Aye!! They stayed. Now there are', len(cust_entered_times), 'people in line.')
                        else:
                            n_custs_who_left += 1
                            print('Rip lol they left. There are still', n, ' people in line.')

                    # Sample the time of next customer's arrivals
                    dt = exp(1./k)
                    total_interarrival_time += dt
                    t_next_cust += dt

                    all_customer_arrival_times.append(t_next_cust)
                    all_interarrival_times.append(dt)

                elif s1.t_when_free == t or s2.t_when_free == t:
                    print('We came to this time step because at least 1 machine freed up. The next customer will arrive at', t_next_cust, '.')

                    # Figure out which machine freed up.
                    for i, s in enumerate({s1, s2}):
                        if s.t_when_free == t:
                            print('In particular, machine', i, ' also known as ', s, 'was freed up.')

                elif t in {s1.break_times[0], s1.break_times[1], s2.break_times[0], s2.break_times[1]}:
                    print('We came to this time step because one of the machines has a scheduled break!')

                else:
                    print('SOMETHING WENT WRONG AND WE SHOULD NOT BE AT THIS TIME STEP!!')

                ''' Check if it is time for a server to take a break. If so, set it to busy (i.e. set its t_when_free to 
                five minutes from now).'''

                for i, s in enumerate({s1, s2}):
                    if t == s.next_break_time():
                        print('Time for Machine', i, ' also known as', s, 'to take a break! It will be free at,', t+5, '.')
                        s.t_when_free = t + 5  # five minute break
                        s.n_breaks_left -= 1

                ''' Process the line: Assign first in line to a server, if possible. NOTE: if a server is free, 
                generate a service time, and if this overlaps with when there will be a break, then shift the break directly
                after the job ends.'''

                s = None  # this variable s will be the server which could potentially be assigned to the
                          # front of the line customer.

                # Case 1) Both are free.
                if s1.t_when_free <= t and s2.t_when_free <= t:
                    print('Both machines are currently free.')
                    # Randomly select which server s to assign the first customer in line.
                    rand = np.random.random()
                    s = s1 if rand <= 0.5 else s2
                    print('We randomly picked machine', s, ' to use for processing the item from ', len(cust_entered_times), 'total on the line.')

                # Case 2) Only one is free.
                elif s1.t_when_free <= t or s2.t_when_free <= t:
                    s = s1 if s1.t_when_free <= t else s2
                    print('Only machine', s, ' was free. So we assigned this one an item from ', len(cust_entered_times), 'total in the line.')

                # Case 3) None are free (meaning we only came to this time step to add in a new customer)
                else:
                    print('None of the machines are free, rip.')
                    pass  # do nothing.

                # END OF CASES –– NOW: If s is a legit server (not 'None') and there is at least 1 waiting customer,
                # then perform the assignment!
                if s is not None and len(cust_entered_times) > 0:
                    # Increment the total waiting time and increment the total NUMBER of waiters only if the time
                    # they entered is NOT identical to the current time (meaning they weren't literally were just added).
                    if cust_entered_times[0] != t:
                        total_waiting_time += t - cust_entered_times[0]
                        n_total_waiters += 1

                    # Update the randomly chosen server's t_when_free, total_service_time, and n_customers_served: using to
                    # the exponential distribution of the server's service time.
                    service_time = exp(1. / s.lamb)
                    s.total_service_time += service_time
                    s.n_customers_served += 1
                    s.t_when_free = t + service_time

                    # Append histogram data lists for service times
                    if s == s1:
                        all_server1_service_times.append(service_time)
                    else:  # if s == s_2
                        all_server2_service_times.append(service_time)

                    # Remove the former first-in-line customer's add-time from the list.
                    cust_entered_times.pop(0)

                    # If a scheduled break has a scheduling conflict with this new service time, then we shift the break
                    # to the end of the service time. Now what may happen is that the shifted break could intersect
                    # with an other scheduled break in the future, so we shift that as well if this is the case.
                    if t < s.next_break_time() < s.t_when_free:
                        print('The machine which was just assigned a job ending at t=', s.t_when_free, 'was scheduled a break at t=', s.next_break_time(), 'but this break was shifted till directly after this job is complete.')
                        s.shift_breaks(s.t_when_free)

                else:
                    print('Actually no items were really assigned.')

                ''' Decide next time step. This time step will either be when a new customer shows up, or when a server
                    frees up, or it is time for a free server to take a break, whichever is earlier. Note that this time step
                    can in fact be larger than 4*60, but in that case the loop will end.'''

                t = min_but_greater_than(t_next_cust, s1.t_when_free, s2.t_when_free, s1.next_break_time(),
                                         s2.next_break_time(), greater_than=t)

                print()

            print('Epoch', epoch, 'complete.')


        '''We output the collected data to answer the questions.'''

        print('done')
        print()
        print('The following was gathered for k =', k, 'after', n_epochs, 'separate trials:')
        print('Average Interval-arrival Time:', total_interarrival_time/total_customers_entered)
        print('Average Service Time By Teller 1:', s1.total_service_time / s1.n_customers_served)
        print('Average Service Time By Teller 2:', s2.total_service_time / s2.n_customers_served)
        print('Proportion of time Teller 1 is busy:', s1.total_service_time / (4*60*n_epochs))
        print('Proportion of time Teller 2 is busy:', s2.total_service_time / (4*60*n_epochs))
        print('Number of customers that had to wait in line:', n_total_waiters)
        print('Average waiting time of customers in line:', total_waiting_time/n_total_waiters if n_total_waiters != 0 else 0)
        print('Number of customers who left:', n_custs_who_left)
        print()

        '''We now create the needed histograms based on the data.'''
        # TODO: Histogram + Plot #1) Number of customers arriving (each hour?), along with Poisson distribution curve
        all_customer_arrival_times  # <-- that is the variable which is a list containing all customer arrival times across 100 epochs


        # TODO: Histogram + Plot #2) Interarrival times, along with the theoretical Exponential distribution
        all_interarrival_times  # <-- that is the variable which is a list containing all interarrival times across 100 epochs


        # TODO: Histogram + Plot #3) Service times, along with its theoretical distribution.
        # Since the service times (time it takes for a certain customer to be processed) does not change with the value
        # of k, we only create a single histogram plot of this kind, say for k=5 (arbitrary choice).
        if k == 5:
            # TODO
            all_server1_service_times  # <-- that is the variable which is a list containing all service times for server 1 across 100 epochs
            all_server2_service_times  # <-- that is the variable which is a list containing all service times for server 2 across 100 epochs


# Modified min function
def min_but_greater_than(*args, greater_than=None):
    '''
        This function returns the smallest out the given arguments (decimal numbers), that is STRICTLY
        greater than the last parameter of the function.

        e.g. min_but_greater_than(3, 4.2, 5, 6, 7, 7.3, greater_than=5) returns 6.
    '''
    new_args = []
    for arg in args:
        if arg > greater_than:
            new_args.append(arg)

    return min(new_args)


# Draw from Exp(lambda)
def exp(lamb):
    '''
        We use the inverse transform method to draw from Exp(lambda), where lambda is the RATE.
    '''

    x = np.random.random()
    return - np.log(1 - x) / lamb



if __name__ == '__main__':
    run()

