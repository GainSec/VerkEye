/*******************************************************************************
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	   See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef __XMAS_LIST__
#define __XMAS_LIST__

struct list_head
{
	struct list_head *prev;
	struct list_head *next;
};

#define container_of(ptr, type, member)		\
	(type *)((char *)ptr - (char *)&(((type *)0)->member))

static void xlist_add(struct list_head *new, struct list_head *prev,
		struct list_head *next)
{
	prev->next = new;
	new->prev = prev;
	new->next = next;
	next->prev = new;
}

static void xlist_del(struct list_head *prev, struct list_head *next)
{
	prev->next = next;
	next->prev = prev;
}

static void inline INIT_LIST_HEAD(struct list_head *head)
{
	head->prev = head;
	head->next = head;
}

static void inline list_add_tail(struct list_head *new, struct list_head *head)
{
	xlist_add(new, head->prev, head);
}

static void inline list_add(struct list_head *new, struct list_head *head)
{
	xlist_add(new, head, head->next);
}

static void inline list_del(struct list_head *entry)
{
	xlist_del(entry->prev, entry->next);
}

static int inline list_empty(struct list_head *entry)
{
	return entry->prev == entry ? 1 : 0;
}

static inline void list_del_init(struct list_head *entry)
{
	list_del(entry);
	INIT_LIST_HEAD(entry);
}
static inline void list_move(struct list_head *new, struct list_head *head)
{
	list_del(new);
	list_add(new, head);
}

static inline void list_move_tail(struct list_head *new,
				  struct list_head *head)
{
	list_del(new);
	list_add_tail(new, head);
}

static void inline xlist_splice(struct list_head *list,
	struct list_head *prev,
	struct list_head *next)
{
	struct list_head *first = list->next;
	struct list_head *last = list->prev;

	first->prev = prev;
	prev->next = first;

	last->next = next;
	next->prev = last;
}

static inline void list_splice(struct list_head *list,
		struct list_head *head)
{
	if (!list_empty(list))
		xlist_splice(list, head, head->next);
}

static inline void list_splice_tail(struct list_head *list,
				struct list_head *head)
{
	if (!list_empty(list))
		xlist_splice(list, head->prev, head);
}

#define list_entry(ptr, type, member)    \
	container_of(ptr, type, member)

#define list_first_entry(head, type, member) \
	list_entry((head)->next, type, member)

#define list_last_entry(head, type, member) \
	list_entry((head)->prev, type, member)

#define list_next_entry(pos, member) \
	list_entry((pos)->member.next, __typeof__(*(pos)), member)

#define list_prev_entry(pos, member) \
	list_entry((pos)->member.prev, __typeof__(*(pos)), member)

#define LIST_HEAD_INIT(name) { &(name), &(name) }

#define LIST_HEAD(name)	\
	struct list_head name = LIST_HEAD_INIT(name)

#define list_for_each(pos, head)	\
	for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_for_each_safe(pos, n, head)				\
	for (pos = (head)->next, n = pos->next; pos != (head);		\
			pos = n, n = pos->next)

#define list_for_each_entry(pos, head, member)				\
	for (pos = list_first_entry(head, __typeof__(*(pos)), member);	\
			&pos->member != head;				\
			pos = list_next_entry(pos, member))

#define list_for_each_entry_safe(pos, n, head, member)			\
	for (pos = list_first_entry(head, typeof(*pos), member),	\
			n = list_next_entry(pos, member);		\
			&pos->member != (head); 			\
			pos = n, n = list_next_entry(n, member))

#define list_for_each_entry_reverse(pos, head, member)			\
	for (pos = list_last_entry(head, typeof(*pos), member);		\
			&pos->member != (head);					\
			pos = list_prev_entry(pos, member))


#endif /* __XMAS_LIST__ */
