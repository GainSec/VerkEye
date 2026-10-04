/*******************************************************************************
 * rundags.c
 *
 * History:
 *    2019/09/11  - [Tao Wu] created
 *
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/errno.h>
#include "cavalry_gen.h"
#include "nnctrl_priv.h"
#include "utils.h"

int set_run_dags_struct(struct net_desc *pnet)
{
	struct cavalry_run_dags *run = NULL;
	dvi_node_t *node = NULL;
	uint32_t i = 0, j = 0, k = 0, m = 0;
	uint32_t phy = 0;
	uint32_t verbose = pnet->verbose;
	int rval = 0;

	run = &pnet->run_dags;
	run->dag_cnt = pnet->header.dvi_num;
	node = pnet->dvi_node_list;
	phy = pnet->phy_addr;

	if (verbose) {
		printf("==== Set run_dags ioctl struct desc START ====\n");
	}

	for (i = 0; i < run->dag_cnt; i++, node++) {
		run->dag_desc[i].dvi_mode = 1;
		run->dag_desc[i].use_ping_pong_vmem = node->dvi_desc.dvi_ppv;
		run->dag_desc[i].dvi_dram_addr = phy + node->dvi_dram_addr;
		run->dag_desc[i].dvi_img_vaddr = node->dvi_desc.dvi_img_vaddr;
		run->dag_desc[i].dvi_img_size  = node->dvi_desc.dvi_img_size;
		run->dag_desc[i].dvi_dag_vaddr = node->dvi_desc.dvi_dag_vaddr;
		run->dag_desc[i].port_cnt =
			node->dvi_desc.input_num + node->dvi_desc.output_num;
		run->dag_desc[i].poke_cnt = 0;
		run->dag_desc[i].dag_loop_cnt = pnet->net_loop_cnt;
		if (run->dag_desc[i].port_cnt > MAX_PORT_CNT) {
			printf("dvi id: %u port cnt too large: %u > %u\n",
				i, run->dag_desc[i].port_cnt, MAX_PORT_CNT);
			rval = -1;
			break;
		}

		j = 0;
		for (k = 0; k < node->dvi_desc.input_num; k++, j++) {
			run->dag_desc[i].port_desc[j].port_dram_addr =
				phy + node->in_port[k].port_dram_addr;
			run->dag_desc[i].port_desc[j].port_boffset_in_dag =
				node->in_port[k].io_desc.port_byte_offset << 3;
			run->dag_desc[i].port_desc[j].port_dram_size = node->in_port[k].port_dram_size;

			if (run->dag_desc[i].dag_loop_cnt > 1) {
				run->dag_desc[i].port_desc[j].port_daddr_increment =
					node->in_port[k].io_desc.port_size;
			} else {
				run->dag_desc[i].port_desc[j].port_daddr_increment = 0;
			}

			if (verbose) {
				printf("Port desc (dvi_id-port_id): %u-%u, name: %s, phys: 0x%x\n",
				i, j, node->in_port[k].io_desc.port_name,
				run->dag_desc[i].port_desc[j].port_dram_addr);
			}

			/* poke rotate-flip */
			if ((node->in_port[k].port_type == PORT_TYPE_INPUT) &&
				(node->in_port[k].io_desc.port_drotate_bit_offset) &&
				(node->in_port[k].rotate_flip_in_dvi !=
				node->in_port[k].rotate_flip_bitmap)) {
				m = run->dag_desc[i].poke_cnt;
				run->dag_desc[i].poke_desc[m].poke_val =
					node->in_port[k].rotate_flip_bitmap;
				run->dag_desc[i].poke_desc[m].poke_vaddr =
					node->in_port[k].io_desc.port_drotate_bit_offset +
					(run->dag_desc[i].dvi_dag_vaddr << 3);
				run->dag_desc[i].poke_desc[m].poke_bsize = ROTATE_FLIP_BIT_NUM;
				run->dag_desc[i].poke_cnt++;
				if (verbose) {
					printf("Poke desc (dvi_id-poke-id): %u-%u, name: %s, poke roate_flip value: 0x%x\n",
						i, m, node->in_port[k].io_desc.port_name, run->dag_desc[i].poke_desc[m].poke_val);
				}
			}

			/* poke pitch */
			if ((node->in_port[k].port_type == PORT_TYPE_INPUT) &&
				(node->in_port[k].update_pitch) &&
				(node->in_port[k].io_desc.port_pitch !=
				node->in_port[k].update_pitch)) {
				m = run->dag_desc[i].poke_cnt;
				run->dag_desc[i].poke_desc[m].poke_val =
					MINUS_ONE(node->in_port[k].update_pitch);
				run->dag_desc[i].poke_desc[m].poke_vaddr = /* port_pitch_offset in byte unit */
					(node->in_port[k].io_desc.port_pitch_offset << 3) +
					(run->dag_desc[i].dvi_dag_vaddr << 3);
				run->dag_desc[i].poke_desc[m].poke_bsize =
					node->in_port[k].io_desc.port_pitch_bsize;
				run->dag_desc[i].poke_cnt++;
				if (verbose) {
					printf("Poke desc (dvi_id-poke-id): %u-%u, name: %s, poke pitch value: 0x%x\n",
						i, m, node->in_port[k].io_desc.port_name, run->dag_desc[i].poke_desc[m].poke_val);
				}
			}
		}

		for (k = 0; k < node->dvi_desc.output_num; k++, j++) {
			run->dag_desc[i].port_desc[j].port_dram_addr =
				phy + node->out_port[k].port_dram_addr;
			run->dag_desc[i].port_desc[j].port_boffset_in_dag =
				node->out_port[k].io_desc.port_byte_offset << 3;
			run->dag_desc[i].port_desc[j].port_dram_size = node->out_port[k].port_dram_size;

			if (run->dag_desc[i].dag_loop_cnt > 1) {
				run->dag_desc[i].port_desc[j].port_daddr_increment =
					node->out_port[k].io_desc.port_size;
			} else {
				run->dag_desc[i].port_desc[j].port_daddr_increment = 0;
			}

			if (verbose) {
				printf("Port desc (dvi_id-port-id): %u-%u, name: %s, phys: 0x%x\n",
					i, j, node->out_port[k].io_desc.port_name,
					run->dag_desc[i].port_desc[j].port_dram_addr);
			}

			/* poke rotate-flip */
			if ((node->out_port[k].port_type == PORT_TYPE_OUTPUT) &&
				(node->out_port[k].io_desc.port_drotate_bit_offset) &&
				(node->out_port[k].rotate_flip_in_dvi !=
				node->out_port[k].rotate_flip_bitmap)) {
				m = run->dag_desc[i].poke_cnt;
				run->dag_desc[i].poke_desc[m].poke_val =
					node->out_port[k].rotate_flip_bitmap;
				run->dag_desc[i].poke_desc[m].poke_vaddr =
					node->out_port[k].io_desc.port_drotate_bit_offset +
					(run->dag_desc[i].dvi_dag_vaddr << 3);
				run->dag_desc[i].poke_desc[m].poke_bsize = ROTATE_FLIP_BIT_NUM;
				run->dag_desc[i].poke_cnt++;
				if (verbose) {
					printf("Poke desc (dvi_id-poke-id): %u-%u, name: %s, poke rotate_flip value: 0x%x\n",
						i, m, node->out_port[k].io_desc.port_name, run->dag_desc[i].poke_desc[m].poke_val);
				}
			}

			/* poke pitch */
			if ((node->out_port[k].port_type == PORT_TYPE_OUTPUT) &&
				(node->out_port[k].update_pitch) &&
				(node->out_port[k].io_desc.port_pitch !=
				node->out_port[k].update_pitch)) {
				m = run->dag_desc[i].poke_cnt;
				run->dag_desc[i].poke_desc[m].poke_val =
					MINUS_ONE(node->out_port[k].update_pitch);
				run->dag_desc[i].poke_desc[m].poke_vaddr = /* port_pitch_offset in byte unit */
					(node->out_port[k].io_desc.port_pitch_offset << 3) +
					(run->dag_desc[i].dvi_dag_vaddr << 3);
				run->dag_desc[i].poke_desc[m].poke_bsize =
					node->out_port[k].io_desc.port_pitch_bsize;
				run->dag_desc[i].poke_cnt++;
				if (verbose) {
					printf("Poke desc (dvi_id-poke-id): %u-%u, name: %s, poke pitch value: 0x%x\n",
						i, m, node->in_port[k].io_desc.port_name, run->dag_desc[i].poke_desc[m].poke_val);
				}
			}
		}
	}
	if (verbose) {
		printf("==== Set run_dags ioctl struct desc END ====\n\n");
	}

	return rval;
}

int update_run_dags_struct_loop_cnt(struct net_desc *pnet,
	struct net_run_cfg *net_run)
{
	struct cavalry_run_dags *run = NULL;
	uint32_t i = 0;

	if (net_run == NULL) {
		return 0;
	}
	if (net_run->net_loop_cnt > pnet->net_loop_cnt) {
		printf("Invalid update net_loop_cnt: %u > %u\n",
			net_run->net_loop_cnt, pnet->net_loop_cnt);
		return -1;
	}

	run = &pnet->run_dags;
	for (i = 0; i < run->dag_cnt; i++) {
		run->dag_desc[i].dag_loop_cnt = net_run->net_loop_cnt;
	}

	return 0;
}

void update_run_dags_struct_port(struct net_desc *pnet)
{
	struct cavalry_run_dags *run = NULL;
	dvi_node_t *node = NULL;
	uint32_t i = 0, j = 0, k = 0;
	uint32_t phy = 0, verbose = 0;

	run = &pnet->run_dags;
	node = pnet->dvi_node_list;
	phy = pnet->phy_addr;
	verbose = pnet->verbose;

	for (i = 0; i < run->dag_cnt; i++, node++) {
		j = 0;
		for (k = 0; k < node->dvi_desc.input_num; k++, j++) {
			if ((node->in_port[k].port_type == PORT_TYPE_INPUT) ||
				(node->in_port[k].port_type == PORT_TYPE_ITM_FROM_OUT)) {
				/* Update port addr */
				run->dag_desc[i].port_desc[j].port_dram_addr =
					phy + node->in_port[k].port_dram_addr;
				if (verbose) {
					printf("Set port struct: %u-%u, port [%s], phys: 0x%x\n",
						i, j, node->in_port[k].io_desc.port_name,
						run->dag_desc[i].port_desc[j].port_dram_addr);
				}
			}
		}

		for (k = 0; k < node->dvi_desc.output_num; k++, j++) {
			if (node->out_port[k].port_type == PORT_TYPE_OUTPUT) {
				/* Update port addr */
				run->dag_desc[i].port_desc[j].port_dram_addr =
					phy + node->out_port[k].port_dram_addr;
				if (verbose) {
					printf("Set port struct: %u-%u, port [%s], phys: 0x%x\n",
						i, j, node->out_port[k].io_desc.port_name,
						run->dag_desc[i].port_desc[j].port_dram_addr);
				}
			}
		}
	}
}

void update_run_dags_struct_poke(struct net_desc *pnet)
{
	struct cavalry_run_dags *run = NULL;
	dvi_node_t *node = NULL;
	uint32_t i = 0, j = 0, k = 0, m = 0;
	uint32_t verbose = 0;

	run = &pnet->run_dags;
	node = pnet->dvi_node_list;
	verbose = pnet->verbose;

	for (i = 0; i < run->dag_cnt; i++, node++) {
		j = 0;

		/* clear poke cnt for some field in init() have but run() have not */
		run->dag_desc[i].poke_cnt = 0;
		for (k = 0; k < node->dvi_desc.input_num; k++, j++) {
			if ((node->in_port[k].port_type == PORT_TYPE_INPUT) ||
				(node->in_port[k].port_type == PORT_TYPE_ITM_FROM_OUT)) {

				/* Update poke rotate_flip */
				if ((node->in_port[k].io_desc.port_drotate_bit_offset) &&
					(node->in_port[k].rotate_flip_in_dvi !=
					node->in_port[k].rotate_flip_bitmap)) {
					m = run->dag_desc[i].poke_cnt;
					run->dag_desc[i].poke_desc[m].poke_val =
						node->in_port[k].rotate_flip_bitmap;
					run->dag_desc[i].poke_desc[m].poke_vaddr =
						node->in_port[k].io_desc.port_drotate_bit_offset +
						(run->dag_desc[i].dvi_dag_vaddr << 3);
					run->dag_desc[i].poke_desc[m].poke_bsize = ROTATE_FLIP_BIT_NUM;
					run->dag_desc[i].poke_cnt++;
					if (verbose) {
						printf("Set poke struct: %u-%u, port [%s] rotate-flip: 0x%x\n",
							i, j, node->in_port[k].io_desc.port_name,
							node->in_port[k].rotate_flip_bitmap);
					}
				}

				/* Update poke pitch */
				if ((node->in_port[k].update_pitch) &&
					(node->in_port[k].io_desc.port_pitch !=
					node->in_port[k].update_pitch)) {
					m = run->dag_desc[i].poke_cnt;
					run->dag_desc[i].poke_desc[m].poke_val =
						MINUS_ONE(node->in_port[k].update_pitch);
					run->dag_desc[i].poke_desc[m].poke_vaddr = /* port_pitch_offset in byte unit */
						(node->in_port[k].io_desc.port_pitch_offset << 3) +
						(run->dag_desc[i].dvi_dag_vaddr << 3);
					run->dag_desc[i].poke_desc[m].poke_bsize =
						node->in_port[k].io_desc.port_pitch_bsize;
					run->dag_desc[i].poke_cnt++;
					if (verbose) {
						printf("Set poke struct: %u-%u, port [%s] update_pitch: %u\n",
							i, j, node->in_port[k].io_desc.port_name,
							node->in_port[k].update_pitch);
					}
				}
			}
		}

		for (k = 0; k < node->dvi_desc.output_num; k++, j++) {
			if (node->out_port[k].port_type == PORT_TYPE_OUTPUT) {
				/* Update poke rotate_flip */
				if ((node->out_port[k].io_desc.port_drotate_bit_offset) &&
					(node->out_port[k].rotate_flip_in_dvi !=
					node->out_port[k].rotate_flip_bitmap)) {
					m = run->dag_desc[i].poke_cnt;
					run->dag_desc[i].poke_desc[m].poke_val =
						node->out_port[k].rotate_flip_bitmap;
					run->dag_desc[i].poke_desc[m].poke_vaddr =
						node->out_port[k].io_desc.port_drotate_bit_offset +
						(run->dag_desc[i].dvi_dag_vaddr << 3);
					run->dag_desc[i].poke_desc[m].poke_bsize = ROTATE_FLIP_BIT_NUM;
					run->dag_desc[i].poke_cnt++;
					if (verbose) {
						printf("Set poke struct: %u-%u, port [%s], rotate-flip: 0x%x\n",
							i, j, node->out_port[k].io_desc.port_name,
							node->out_port[k].rotate_flip_bitmap);
					}
				}

				/* Update poke pitch */
				if ((node->out_port[k].update_pitch) &&
					(node->out_port[k].io_desc.port_pitch !=
					node->out_port[k].update_pitch)) {
					m = run->dag_desc[i].poke_cnt;
					run->dag_desc[i].poke_desc[m].poke_val =
						MINUS_ONE(node->out_port[k].update_pitch);
					run->dag_desc[i].poke_desc[m].poke_vaddr = /* port_pitch_offset in byte unit */
						(node->out_port[k].io_desc.port_pitch_offset << 3) +
						(run->dag_desc[i].dvi_dag_vaddr << 3);
					run->dag_desc[i].poke_desc[m].poke_bsize =
						node->out_port[k].io_desc.port_pitch_bsize;
					run->dag_desc[i].poke_cnt++;
					if (verbose) {
						printf("Set poke struct: %u-%u, port [%s] update_pitch: %u\n",
							i, j, node->out_port[k].io_desc.port_name,
							node->out_port[k].update_pitch);
					}
				}
			}
		}
	}
}

int split_net_run(struct nnctrl_info *pctl, struct net_desc *pnet, int net_id, uint32_t split_num,
	float *vp_time_us)
{
	struct cavalry_run_dags *org = NULL;
	struct cavalry_run_dags *run = NULL;
	struct timeval tv1, tv2;
	unsigned long tv_diff = 0;
	unsigned long diff_tick = 0, total_tick = 0;
	float singl_vp_us = 0.0f;
	uint32_t nfinished = 0, npartial = 0;
	uint32_t total = 0, n = 0, last_idx = 0;
	int rval = 0;

	run = &pnet->partial_dags;
	org = &pnet->run_dags;
	total = org->dag_cnt;
	nfinished = 0;
	npartial = total /split_num;
	last_idx = npartial * (split_num - 1);
	if (!npartial) {
		printf("Invalid partial num: %u = (%u/%u) in one slice of network\n",
			npartial, total, split_num);
		return -1;
	}

	if (pnet->print_time) {
		gettimeofday(&tv1, NULL);
	}
	while (nfinished < total) {
		run->rval = 0;
		run->start_tick = 0;
		run->end_tick = 0;
		run->finish_dags = 0;
		if (nfinished >= last_idx) {
			n = total - nfinished;
		} else {
			n = npartial;
		}
		run->dag_cnt = n;
		memcpy(&run->dag_desc[0], &org->dag_desc[nfinished],
			n * sizeof(struct cavalry_dag_desc));

		if (ioctl(pctl->fd_cav, CAVALRY_RUN_DAGS, run) < 0) {
			if (errno == EINTR) {
				rval = -EINTR;
			} else {
				perror("CAVALRY_RUN_DAGS");
				rval = -1;
			}
		}

		diff_tick = run->start_tick <= run->end_tick ? (run->end_tick - run->start_tick) :
			(0xFFFFFFFF - run->start_tick + run->end_tick);
		singl_vp_us = (float)diff_tick / pctl->audio_clk_mhz;
		total_tick += diff_tick;

		if (rval == 0) {
			nfinished += run->finish_dags;
			if (run->finish_dags != run->dag_cnt) {
				if (run->finish_dags < run->dag_cnt) {
					pnet->dag_idx = nfinished;
					rval = -EAGAIN;
				} else {
					printf("Abnormal run dags: %u / %u\n", run->finish_dags, run->dag_cnt);
					rval = -1;
				}
			}
		}
		if (pnet->print_time) {
			printf("Net_id: %d, Dag_cnt: %u, vp_ticks: %lu, vp_time: %lu us. [Run Split]\n",
				net_id, run->finish_dags, diff_tick, (unsigned long)singl_vp_us);
		}
		if (rval < 0) {
			break;
		}
	}
	if (pnet->print_time) {
		gettimeofday(&tv2, NULL);
	}

	*vp_time_us = total_tick / pctl->audio_clk_mhz;
	if (pnet->print_time) {
		tv_diff = (unsigned long) 1000000 * (unsigned long) (tv2.tv_sec - tv1.tv_sec) +
			(unsigned long) (tv2.tv_usec - tv1.tv_usec);
		printf("Net_id: %d, Dags: %u / %u, vp_ticks: %lu, vp_time: %lu us, arm_time: %lu us\n",
			net_id, nfinished, org->dag_cnt, total_tick, (unsigned long)*vp_time_us, tv_diff);
	}

	return rval;
}

int single_dag_run(struct nnctrl_info *pctl, struct net_desc *pnet, int net_id,
	float *vp_time_us)
{
	struct cavalry_run_dags *org = NULL;
	struct cavalry_run_dags *run = NULL;
	struct timeval tv1, tv2;
	unsigned long tv_diff = 0;
	unsigned long diff_tick = 0, total_tick = 0;
	float singl_vp_us = 0.0f;
	uint32_t i = 0;
	int rval = 0;

	run = &pnet->partial_dags;
	org = &pnet->run_dags;

	if (pnet->print_time) {
		gettimeofday(&tv1, NULL);
	}
	for (i = 0; i < org->dag_cnt; ) {
		run->rval = 0;
		run->start_tick = 0;
		run->end_tick = 0;
		run->finish_dags = 0;
		run->dag_cnt = 1;
		run->dag_desc[0] = org->dag_desc[i];

		if (ioctl(pctl->fd_cav, CAVALRY_RUN_DAGS, run) < 0) {
			if (errno == EINTR) {
				rval = -EINTR;
			} else {
				perror("CAVALRY_RUN_DAGS");
				rval = -1;
			}
		}

		diff_tick = run->start_tick <= run->end_tick ? (run->end_tick - run->start_tick) :
			(0xFFFFFFFF - run->start_tick + run->end_tick);
		singl_vp_us = (float)diff_tick / pctl->audio_clk_mhz;
		total_tick += diff_tick;

		if (rval == 0) {
			if (run->finish_dags != run->dag_cnt) {
				if (run->finish_dags > run->dag_cnt) {
					printf("Abnormal single run dag: %u / %u\n", run->finish_dags, run->dag_cnt);
					rval = -1;
				}
			}
			i += run->finish_dags;
		}
		if (pnet->print_time) {
			printf("Net_id: %d, Dag_id: %u, vp_ticks: %lu, vp_time: %lu us. [Run Single Dag]\n",
				net_id, i, diff_tick, (unsigned long)singl_vp_us);
		}
		if (rval < 0) {
			break;
		}
	}
	if (pnet->print_time) {
		gettimeofday(&tv2, NULL);
	}

	*vp_time_us = total_tick / pctl->audio_clk_mhz;
	if (pnet->print_time) {
		tv_diff = (unsigned long) 1000000 * (unsigned long) (tv2.tv_sec - tv1.tv_sec) +
			(unsigned long) (tv2.tv_usec - tv1.tv_usec);
		printf("Net_id: %d, Dags: %u / %u, vp_ticks: %lu, vp_time: %lu us, arm_time: %lu us\n",
			net_id, i, org->dag_cnt, total_tick, (unsigned long)*vp_time_us, tv_diff);
	}

	return rval;
}

int all_dag_run(struct nnctrl_info *pctl, struct net_desc *pnet, int net_id,
	float *vp_time_us)
{
	struct cavalry_run_dags *run = NULL;
	struct timeval tv1, tv2;
	unsigned long tv_diff = 0;
	unsigned long diff_tick = 0;
	int rval = 0;

	run = &pnet->run_dags;
	run->rval = 0;
	run->start_tick = 0;
	run->end_tick = 0;
	run->finish_dags = 0;

	if (pnet->print_time) {
		gettimeofday(&tv1, NULL);
	}
	if (ioctl(pctl->fd_cav, CAVALRY_RUN_DAGS, run) < 0) {
		if (errno == EINTR) {
			rval = -EINTR;
		} else {
			perror("CAVALRY_RUN_DAGS");
			rval = -1;
		}
	}
	if (pnet->print_time) {
		gettimeofday(&tv2, NULL);
	}
	diff_tick = run->start_tick <= run->end_tick ? (run->end_tick - run->start_tick) :
		(0xFFFFFFFF - run->start_tick + run->end_tick);
	*vp_time_us = (float)diff_tick / pctl->audio_clk_mhz;
	if (rval == 0) {
		if (run->finish_dags != run->dag_cnt) {
			if (run->finish_dags < run->dag_cnt) {
				pnet->dag_idx = run->finish_dags;
				rval = -EAGAIN;
			} else {
				printf("Abnormal run dags: %u / %u\n", run->finish_dags, run->dag_cnt);
				rval = -1;
			}
		}
	}

	if (pnet->print_time) {
		tv_diff = (unsigned long) 1000000 * (unsigned long) (tv2.tv_sec - tv1.tv_sec) +
			(unsigned long) (tv2.tv_usec - tv1.tv_usec);
		printf("Net_id: %d, Dags: %u / %u, vp_ticks: %lu, vp_time: %lu us, arm_time: %lu us\n",
			net_id, run->finish_dags, run->dag_cnt, diff_tick, (unsigned long)*vp_time_us, tv_diff);
	}

	return rval;
}

