/*******************************************************************************
 * parser.c
 *
 * History:
 *    2018/08/22  - [Tao Wu] created
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
#include "cavalry_gen.h"
#include "nnctrl_priv.h"
#include "utils.h"

#define API_MASK (0xffff)

int check_cavalry_gen_bin_header(struct net_desc *pnet)
{
	cavalry_gen_header_t header;

	if (pnet->net_fp) {
		if (fread(&header, sizeof(header), 1, pnet->net_fp) != 1) {
			perror("fread cavalry head");
			return -1;
		}
	} else {
		memcpy(&header, pnet->net_feed_virt + pnet->net_feed_virt_pos, sizeof(header));
		pnet->net_feed_virt_pos += sizeof(header);
	}

	if ((header.version_info & API_MASK) != (CAVALRY_GEN_VERSION & API_MASK)) {
		printf("cavalry_gen binary version [0x%08x] != parser version [0x%08x]\n",
			header.version_info & API_MASK, CAVALRY_GEN_VERSION & API_MASK);
		printf("Please update <NET>.bin with new cavalry_gen tool\n");
		return -1;
	}
	if (header.dvi_num > MAX_DAG_CNT) {
		printf("dvi number in binary too large: %u > %u\n", header.dvi_num, MAX_DAG_CNT);
		return -1;
	}
	pnet->header = header;

	return 0;
}

int check_io_num(struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	if (net_in && (net_in->in_num > MAX_IO_NUM)) {
		printf("Net In num %u exceed max: %u\n", net_in->in_num, MAX_IO_NUM);
		return -1;
	}
	if (net_out && (net_out->out_num > MAX_IO_NUM)) {
		printf("Net In num %u exceed max: %u\n", net_out->out_num, MAX_IO_NUM);
		return -1;
	}

	return 0;
}

static int get_dvi_desc(struct net_desc *pnet, dvi_desc_t *dvi_desc)
{
	if (pnet->net_fp) {
		if (fread(dvi_desc, sizeof(dvi_desc_t), 1, pnet->net_fp) != 1) {
			perror("fread dvi desc");
			return -1;
		}
	} else {
		memcpy(dvi_desc, pnet->net_feed_virt + pnet->net_feed_virt_pos, sizeof(dvi_desc_t));
		pnet->net_feed_virt_pos += sizeof(dvi_desc_t);
	}

	return 0;
}

static int allocate_dvi_mem(struct net_desc *pnet, dvi_node_t *node)
{
	long pos = 0;
	int rval = 0;

	if (pnet->verbose) {
		printf("allocate mem dvi id: %u, size: %u\n",
			node->dvi_desc.dvi_id, node->dvi_desc.dvi_img_size);
	}

	node->dvi_dram_addr = layout_mem(pnet, node->dvi_desc.dvi_img_size);
	pnet->dvi_mem_total += node->dvi_desc.dvi_img_size;

	do {
		if (pnet->net_fp) {
			pos = ftell(pnet->net_fp);
			if (pos < 0) {
				perror("ftell");
				rval = -1;
				break;
			}
		} else {
			pos = pnet->net_feed_virt_pos;
		}
		node->dvi_file_pos = pos;

		if (pnet->net_fp) {
			if (fseek(pnet->net_fp, node->dvi_desc.dvi_img_size, SEEK_CUR) < 0) {
				perror("fseek");
				rval = -1;
				break;
			}
		} else {
			pnet->net_feed_virt_pos += node->dvi_desc.dvi_img_size;
		}
	} while (0);

	pnet->dvi_mem_align_total = pnet->mem_offset;

	return rval;
}

static int allocate_port_mem_reuse(struct net_desc *pnet,
	port_desc_t *curr_port, uint32_t is_input)
{
	/* search for previous re-usable port memory, assign pointer */
	dvi_node_t *head_node = NULL, *node = NULL;
	uint32_t dvi_id = 0, found = 0;
	uint32_t i = 0, j = 0, k = 0;
	int rval = 0;

	if (curr_port->port_type != PORT_TYPE_ITM) {
		//printf("Reuse skip port [%s] mem\n", curr_port->io_desc.port_name);
		return 0;
	}

	dvi_id = curr_port->dvi_id;
	head_node = pnet->dvi_node_list;
	node = head_node;
	if (is_input) {
		if (curr_port->port_type == PORT_TYPE_ITM) {
			for (i = 0; i < dvi_id; i++) {
				for (j = 0; j < node[i].dvi_desc.input_num; j++) {
					/* if share input with another DAG */
					if (strcmp(curr_port->io_desc.port_name,
							node[i].in_port[j].io_desc.port_name) == 0) {
						found = 1;
						curr_port->port_dram_addr = node[i].in_port[j].port_dram_addr;
						break;
					}
				}
				if (found) {
					break;
				}

				for (j = 0; j < node[i].dvi_desc.output_num; j++) {
					/* if needs the output of another DAG */
					if (strcmp(curr_port->io_desc.port_name,
							node[i].out_port[j].io_desc.port_name) == 0) {
						found = 1;
						curr_port->port_dram_addr = node[i].out_port[j].port_dram_addr;
						break;
					}
				}
				if (found) {
					break;
				}
			}
			if (!found) {
				printf("internal bug. dvi[%u] input port [%s] is of type ITM, "
						"but could not find it's correspondence.\n",
						dvi_id, curr_port->io_desc.port_name);
				rval = -1;
			}
		}
	} else {
		for (i = 0; i < dvi_id; i++) {
			for (j = 0; j < node[i].dvi_desc.output_num; j++) {
				/* search for re-usable output buffer */
				if (node[i].out_port[j].port_remain_size >= curr_port->port_dram_size) {
					if (node[i].out_port[j].needed_by_dvi_cnt > 0) {
						for (k = 0; k < node[i].out_port[j].needed_by_dvi_cnt; k++) {
							if (node[i].out_port[j].needed_by_dvi[k] >= dvi_id) {
								break;
							}
						}
						if (k == node[i].out_port[j].needed_by_dvi_cnt) {
							found = 1;
							curr_port->port_dram_addr =
								node[i].out_port[j].port_dram_addr +
								node[i].out_port[j].port_dram_size -
								node[i].out_port[j].port_remain_size;
							curr_port->port_remain_size = curr_port->port_dram_size;
							node[i].out_port[j].port_remain_size -= curr_port->port_dram_size;
						}
					}
				}
				if (found) {
					break;
				}
			}
			if (found) {
				break;
			}
		}

		/* allocate buffer for output */
		if (!found) {
			if (pnet->verbose) {
				printf("allocate mem ITM port: %s, size: %u\n",
					curr_port->io_desc.port_name, curr_port->port_dram_size);
			}
			if (curr_port->no_mem == 0) {
				curr_port->port_dram_addr = layout_mem(pnet, curr_port->port_dram_size);
				curr_port->port_remain_size = curr_port->port_dram_size;
				pnet->blob_mem_total += curr_port->port_dram_size;
			}
		}
	}

	return rval;
}

static int allocate_port_mem_naive(struct net_desc *pnet,
	port_desc_t *curr_port, uint32_t is_input)
{
	/* Compare with all previous port, allocate buffer or assign pointer */
	dvi_node_t *head_node = NULL, *node = NULL;
	port_desc_t *port = NULL;
	uint32_t dvi_id = 0, found = 0;
	uint32_t i = 0, j = 0;
	int rval = 0;

	if (curr_port->port_type != PORT_TYPE_ITM) {
		//printf("Naive skip port [%s] mem\n", curr_port->io_desc.port_name);
		return 0;
	}

	dvi_id = curr_port->dvi_id;
	head_node = pnet->dvi_node_list;
	node = head_node;
	for (i = 0; i < dvi_id; i++) {
		port = node->in_port;
		for (j = 0; j < node->dvi_desc.input_num; j++) {
			if (strcmp(curr_port->io_desc.port_name, port->io_desc.port_name) == 0) {
				found = 1;
				curr_port->port_dram_addr = port->port_dram_addr;
				if (!is_input) {
					printf("dvi_id[%u] output port [%s] points to "
						"dvi_id[%u]'s input port [%s]."
						"Recursive NN not supported yet."
						"But now use the former one's memory.\n",
						dvi_id, curr_port->io_desc.port_name,
						i, port->io_desc.port_name);
					rval = -1;
				}
				break;
			}
			port++;
		}
		if (found) {
			break;
		}

		port = node->out_port;
		for (j = 0; j < node->dvi_desc.output_num; j++) {
			if (strcmp(curr_port->io_desc.port_name, port->io_desc.port_name) == 0) {
				found = 1;
				curr_port->port_dram_addr = port->port_dram_addr;
				if (!is_input) {
					printf("dvi_id[%u] output port [%s] points to "
						"dvi_id[%u]'s output port [%s]. Please verify!"
						"But now use the former one's memory.\n",
						dvi_id, curr_port->io_desc.port_name,
						i, port->io_desc.port_name);
					rval = -1;
				}
				break;
			}
			port++;
		}
		if (found) {
			break;
		}

		node++;
	}

	if (!found) {
		if (pnet->verbose) {
			printf("allocate mem ITM port: %s, size: %d\n",
				curr_port->io_desc.port_name, curr_port->port_dram_size);
		}
		if (curr_port->no_mem == 0) {
			curr_port->port_dram_addr = layout_mem(pnet, curr_port->port_dram_size);
			pnet->blob_mem_total += curr_port->port_dram_size;
			pnet->blob_name[pnet->blob_num] = curr_port->io_desc.port_name;
			pnet->blob_size[pnet->blob_num] = curr_port->port_dram_size;
			pnet->blob_addr[pnet->blob_num] = curr_port->port_dram_addr;
			pnet->blob_num++;
		}
	}

	return rval;
}

static void allocate_net_in_mem(struct net_desc *pnet)
{
	parent_port_desc_t *prt_port = NULL;
	port_desc_t **port = NULL;
	uint32_t i = 0, j = 0, prt_num = 0;
	uint32_t addr = 0;

	prt_num = pnet->net_prt_in_num;
	prt_port = pnet->net_prt_in;
	for (i = 0; i < prt_num; i++, prt_port++) {
		if (prt_port->no_mem) {
			continue;
		}
		if (pnet->verbose) {
			printf("allocate INPUT: %s, dram_size: %d\n",
				prt_port->name, prt_port->port_dram_size);
		}
		prt_port->port_dram_fd = pnet->mem_fd;
		prt_port->port_dram_addr = layout_mem(pnet, prt_port->port_dram_size);
		prt_port->port_remain_size = prt_port->port_dram_size;
		pnet->blob_mem_total += prt_port->port_dram_size;

		/* split parent addr to sub addr */
		addr = prt_port->port_dram_addr;
		port = prt_port->sub_port;
		for (j = 0; j < prt_port->sub_port_num; j++) {
			port[j]->port_dram_addr = addr;
			port[j]->port_remain_size = port[j]->port_dram_size;
			addr += port[j]->port_dram_size;

			pnet->blob_name[pnet->blob_num] = port[j]->io_desc.port_name;
			pnet->blob_size[pnet->blob_num] = port[j]->port_dram_size;
			pnet->blob_addr[pnet->blob_num] = port[j]->port_dram_addr;
			pnet->blob_num++;
		}
	}
}

static void allocate_net_out_mem(struct net_desc *pnet)
{
	parent_port_desc_t *prt_port = NULL;
	port_desc_t **port = NULL;
	dvi_node_t *node = NULL;
	uint32_t i = 0, j = 0, k = 0, n = 0;
	uint32_t addr = 0, found = 0, dvi_id =0, prt_num = 0;

	prt_num = pnet->net_prt_out_num;
	prt_port = pnet->net_prt_out;
	for (n = 0; n < prt_num; n++, prt_port++) {
		if (prt_port->no_mem) {
			continue;
		}
		prt_port->port_dram_fd = pnet->mem_fd;
		if (pnet->reuse_mem == 0) {
			if (pnet->verbose) {
				printf("allocate OUTPUT: %s, dram_size: %d\n",
					prt_port->name, prt_port->port_dram_size);
			}
			prt_port->port_dram_addr = layout_mem(pnet, prt_port->port_dram_size);
			prt_port->port_remain_size = prt_port->port_dram_size;
			pnet->blob_mem_total += prt_port->port_dram_size;
		} else {
			dvi_id = prt_port->dvi_id;
			node = pnet->dvi_node_list;
			found = 0;
			for (i = 0; i < dvi_id; i++) {
				for (j = 0; j < node[i].dvi_desc.output_num; j++) {
					/* search for re-usable output buffer */
					if (node[i].out_port[j].port_remain_size >= prt_port->port_dram_size) {
						if (node[i].out_port[j].needed_by_dvi_cnt > 0) {
							for (k = 0; k < node[i].out_port[j].needed_by_dvi_cnt; k++) {
								if (node[i].out_port[j].needed_by_dvi[k] >= dvi_id) {
									break;
								}
							}
							if (k == node[i].out_port[j].needed_by_dvi_cnt) {
								found = 1;
								prt_port->port_dram_addr =
									node[i].out_port[j].port_dram_addr +
									node[i].out_port[j].port_dram_size -
									node[i].out_port[j].port_remain_size;
								prt_port->port_remain_size = prt_port->port_dram_size;
								node[i].out_port[j].port_remain_size -= prt_port->port_dram_size;
							}
						}
					}
					if (found) {
						break;
					}
				}
				if (found) {
					break;
				}
			}
			/* allocate buffer for output */
			if (!found) {
				if (pnet->verbose) {
					printf("allocate OUTPUT: %s, dram_size: %d\n",
						prt_port->name, prt_port->port_dram_size);
				}
				prt_port->port_dram_addr = layout_mem(pnet, prt_port->port_dram_size);
				prt_port->port_remain_size = prt_port->port_dram_size;
				pnet->blob_mem_total += prt_port->port_dram_size;
			}
		}

		/* split parent addr to sub addr */
		addr = prt_port->port_dram_addr;
		port = prt_port->sub_port;
		for (j = 0; j < prt_port->sub_port_num; j++) {
			port[j]->port_dram_addr = addr;
			port[j]->port_remain_size = port[j]->port_dram_size;
			addr += port[j]->port_dram_size;

			pnet->blob_name[pnet->blob_num] = port[j]->io_desc.port_name;
			pnet->blob_size[pnet->blob_num] = port[j]->port_dram_size;
			pnet->blob_addr[pnet->blob_num] = port[j]->port_dram_addr;
			pnet->blob_num++;
		}
	}
}

static int chain_itm_from_out_addr(struct net_desc *pnet, port_desc_t *curr_port)
{
	dvi_node_t *node = NULL;
	port_desc_t *port = NULL;
	uint32_t i = 0, j = 0, found = 0;
	int rval = 0;

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		port = node->out_port;
		for (j = 0; j < node->dvi_desc.output_num; j++, port++) {
			if (port->port_type == PORT_TYPE_OUTPUT) {
				if (!strcmp(curr_port->io_desc.port_name, port->io_desc.port_name)) {
					found = 1;
					curr_port->port_dram_addr = port->port_dram_addr;
					if (pnet->verbose) {
						printf("Chain ITM_FRM_OUT: %s to addr: 0x%x\n",
							curr_port->io_desc.port_name, curr_port->port_dram_addr);
					}
					break;
				}
			}
		}
		if (found) {
			break;
		}
	}
	if (!found) {
		rval = -1;
	}

	return rval;
}

static int allocate_itm_from_out_mem(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	port_desc_t *port = NULL;
	uint32_t i = 0, j = 0;
	int rval = 0;

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		port = node->in_port;
		for (j = 0; j < node->dvi_desc.input_num; j++, port++) {
			if (port->port_type == PORT_TYPE_ITM_FROM_OUT) {
				rval = chain_itm_from_out_addr(pnet, port);
				if (rval < 0) {
					printf("chain_itm_from_out_addr err\n");
					break;
				}
			}
		}
		if (rval < 0) {
			break;
		}
	}

	return rval;
}

int allocate_port_mem(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	uint32_t blob_mem_align_start = 0;
	uint32_t i = 0, j = 0;
	int rval = 0;

	blob_mem_align_start = pnet->mem_offset;

	allocate_net_in_mem(pnet);

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++) {
		for (j = 0; j < node[i].dvi_desc.input_num; j++) {
			if (pnet->reuse_mem) {
				rval = allocate_port_mem_reuse(pnet, &node[i].in_port[j], 1);
			} else {
				rval = allocate_port_mem_naive(pnet, &node[i].in_port[j], 1);
			}
			if (rval < 0) {
				printf("allocate dvi id: %u with input port: %s err\n",
					i, node[i].in_port[j].io_desc.port_name);
				rval = -1;
				break;
			}
			pnet->blob_bw_total += node[i].in_port[j].port_dram_size;
		}
		if (rval < 0) {
			break;
		}

		for (j = 0; j < node[i].dvi_desc.output_num; j++) {
			if (pnet->reuse_mem) {
				rval = allocate_port_mem_reuse(pnet, &node[i].out_port[j], 0);
			} else {
				rval = allocate_port_mem_naive(pnet, &node[i].out_port[j], 0);
			}
			if (rval < 0) {
				printf("allocate dvi id: %u with output port: %s err\n",
					i, node[i].out_port[j].io_desc.port_name);
				rval = -1;
				break;
			}
			pnet->blob_bw_total += node[i].out_port[j].port_dram_size;
		}
		if (rval < 0) {
			break;
		}
	}

	if (!rval) {
		allocate_net_out_mem(pnet);

		rval = allocate_itm_from_out_mem(pnet);

		pnet ->blob_mem_align_total = pnet->mem_offset - blob_mem_align_start;
	}

	return rval;
}

static int build_port_dependency(dvi_node_t *head_node, port_desc_t *curr_port)
{
	/* curr_port should be one DAG's input port
	 * Compare with all previous port, decide DAG dependency.
	 * Decide buffer type. */
	dvi_node_t *node = NULL;
	port_desc_t *port = NULL;
	uint32_t dvi_id = 0;
	uint32_t i = 0, j = 0, k = 0;
	uint32_t found = 0;
	int rval = 0;

	dvi_id = curr_port->dvi_id;
	node = head_node;
	for (i = 0; i < dvi_id; i++, node++) {
		port = node->in_port;
		for (j = 0; j < node->dvi_desc.input_num; j++, port++) {
			if (strcmp(curr_port->io_desc.port_name, port->io_desc.port_name) == 0) {
				/* share input with another DAG */
				found = 1;
				/* Do not change port type when set is_primary */
				if (!curr_port->io_desc.port_is_main_io) {
					curr_port->port_type = PORT_TYPE_ITM;
				}
				break;
			}
		}
		if (found) {
			break;
		}

		port = node->out_port;
		for (j = 0; j < node->dvi_desc.output_num; j++, port++) {
			if (strcmp(curr_port->io_desc.port_name, port->io_desc.port_name) == 0) {
				/* needs the output of another DAG */
				found = 1;
				for (k = 0; k < port->needed_by_dvi_cnt; k++) {
					if (port->needed_by_dvi[k] == dvi_id) {
						break;
					}
				}
				if (k == port->needed_by_dvi_cnt) {
					if (port->needed_by_dvi_cnt >= MAX_DEPENDENCY_NUM) {
						printf("port: %s has too many dependency: %u >= %u\n",
							port->io_desc.port_name, port->needed_by_dvi_cnt,
							 MAX_DEPENDENCY_NUM);
						rval = -1;
						break;
					}
					port->needed_by_dvi[port->needed_by_dvi_cnt] = dvi_id;
					port->needed_by_dvi_cnt += 1;
				}

				/* Do not change port type when set is_primary */
				if (!port->io_desc.port_is_main_io) {
					port->port_type = PORT_TYPE_ITM;
				}
				if (!curr_port->io_desc.port_is_main_io) {
					curr_port->port_type = PORT_TYPE_ITM;
				}
				/* itm port is from network's output */
				if ((port->port_type == PORT_TYPE_OUTPUT) &&
					(curr_port->port_type == PORT_TYPE_ITM)) {
					curr_port->port_type = PORT_TYPE_ITM_FROM_OUT;
				}

				break;
			}
		}
		if (rval < 0) {
			break;
		}
		if (found) {
			break;
		}
	}

	return rval;
}

static int check_io_desc(io_descriptor_t *io_desc)
{
	int rval = 0;

	do {
		if (io_desc->port_data_size == DATASIZE_INVALID) {
			printf("Input [%s] get invalid data_format.size\n", io_desc->port_name);
			rval = -1;
			break;
		}
		if (!io_desc->port_dram_format) {
			if (!io_desc->port_dim_bitvector) {
				if ((io_desc->port_dim_w * (1 << io_desc->port_data_size)) >
					(io_desc->port_pitch)) {
					printf("Port [%s] get invalid width/pitch: %u*(1<<%u) > %u\n",
						io_desc->port_name, io_desc->port_dim_w,
						io_desc->port_data_size, io_desc->port_pitch);
					rval = -1;
					break;
				}
			} else {
				if ((io_desc->port_dim_w * (1 << io_desc->port_data_size) / 8) >
					(io_desc->port_pitch)) {
					printf("Port [%s] get invalid width/pitch for bitvector: %u*(1<<%u)/8 > %u\n",
						io_desc->port_name, io_desc->port_dim_w,
						io_desc->port_data_size, io_desc->port_pitch);
					rval = -1;
					break;
				}
			}
		}
		if (io_desc->port_slice_total_num > MAX_SUB_PORT_NUM) {
			printf("Port [%s] has too much slice_num: %u > %u\n", io_desc->port_name,
				io_desc->port_slice_total_num, MAX_SUB_PORT_NUM);
			rval = -1;
			break;
		}
	} while (0);

	return rval;
}

static void set_rotate_flip_bitmap(port_desc_t *port)
{
	uint8_t bitmap = 0, onebit  = 0;

	if (port->io_desc.port_drotate_bit_offset) {
		onebit = port->io_desc.port_drotate;
		bitmap |= (onebit << DROTATE_BIT);
		onebit = port->io_desc.port_hflip;
		bitmap |= (onebit << HFLIP_BIT);
		onebit = port->io_desc.port_vflip;
		bitmap |= (onebit << VFLIP_BIT);
		onebit = port->io_desc.port_dflip;
		bitmap |= (onebit << DFLIP_BIT);
		onebit = port->io_desc.port_pflip;
		bitmap |= (onebit << PFLIP_BIT);

		port->rotate_flip_in_dvi = bitmap;
	}
}

static int gen_net_sub_port(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	port_desc_t *port = NULL;
	uint32_t i = 0, j = 0;

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		for (j = 0; j < node->dvi_desc.input_num; j++) {
			port = &node->in_port[j];
			if (port->port_type == PORT_TYPE_INPUT) {
				if (check_io_desc(&port->io_desc) < 0) {
					return -1;
				}
				pnet->net_in[pnet->net_in_num] = port;
				pnet->net_in_num++;
				set_rotate_flip_bitmap(port);
			}
		}

		for (j = 0; j < node->dvi_desc.output_num; j++) {
			port = &node->out_port[j];
			if (port->port_type == PORT_TYPE_OUTPUT) {
				if (check_io_desc(&port->io_desc) < 0) {
					return -1;
				}
				pnet->net_out[pnet->net_out_num] = port;
				pnet->net_out_num++;
				set_rotate_flip_bitmap(port);
			}
		}
	}

	if (pnet->verbose) {
		for (j = 0; j < pnet->net_in_num; j++) {
			printf("Sub Input: %s\n", pnet->net_in[j]->io_desc.port_name);
		}
		for (j = 0; j < pnet->net_out_num; j++) {
			printf("Sub Output: %s\n", pnet->net_out[j]->io_desc.port_name);
		}
	}

	return 0;
}

static int gen_net_parent_port(struct net_desc *pnet)
{
	port_desc_t **port = NULL;
	uint32_t i = 0, k = 0;
	uint32_t slice_total_num = 0;
	uint32_t found_num = 0, port_num = 0, prt_idx = 0;
	int rval = 0;

	/* net input */
	prt_idx = 0;
	port = pnet->net_in;
	port_num = pnet->net_in_num;
	for (i = 0; i < port_num; i++, prt_idx++) {
		if (prt_idx >= MAX_IO_NUM) {
			printf("Net have too much input: %u >= %u\n", prt_idx, MAX_IO_NUM);
			rval = -1;
			break;
		}
		if (port[i]->io_desc.port_slice_total_num <= 1) {
			if (strlen(port[i]->io_desc.port_demangled_name) > 0) {
				pnet->net_prt_in[prt_idx].name = port[i]->io_desc.port_demangled_name;
			} else {
				pnet->net_prt_in[prt_idx].name = port[i]->io_desc.port_name;
			}
			pnet->net_prt_in[prt_idx].sub_port[pnet->net_prt_in[prt_idx].sub_port_num] = port[i];
			pnet->net_prt_in[prt_idx].sub_port_num++;
		} else {
			if (port[i]->io_desc.port_slice_seq != 0) {
				printf("the first slice_seq should be zero, but it is %d\n",
					port[i]->io_desc.port_slice_seq);
				rval = -1;
				break;
			}
			found_num = 1;
			pnet->net_prt_in[prt_idx].name = port[i]->io_desc.port_slice_parent_name;
			pnet->net_prt_in[prt_idx].sub_port[pnet->net_prt_in[prt_idx].sub_port_num] = port[i];
			pnet->net_prt_in[prt_idx].sub_port_num++;
			slice_total_num = port[i]->io_desc.port_slice_total_num;

			while ((found_num < slice_total_num) && (i < port_num)) {
				i++;
				if (!strcmp(port[i]->io_desc.port_slice_parent_name,
					pnet->net_prt_in[prt_idx].name)) {
					//printf("input [%s] [%s]\n", port[i]->io_desc.port_name,
					//	port[i]->io_desc.port_slice_parent_name);
					if (port[i]->port_type != PORT_TYPE_INPUT) {
						printf("port_type should be input, but it is %d\n",
							port[i]->port_type);
						rval = -1;
						break;
					}
					if (port[i]->io_desc.port_slice_seq != found_num) {
						printf("slice_seq is wrong, %d, should be %d\n",
							port[i]->io_desc.port_slice_seq, k);
						rval = -1;
						break;
					}
					pnet->net_prt_in[prt_idx].sub_port[pnet->net_prt_in[prt_idx].sub_port_num] = port[i];
					pnet->net_prt_in[prt_idx].sub_port_num++;
					found_num++;
				}
			}
			if (rval < 0) {
				break;
			}
		}
	}
	pnet->net_prt_in_num = prt_idx;
	if (rval < 0) {
		return rval;
	}

	/* net output */
	prt_idx = 0;
	port = pnet->net_out;
	port_num = pnet->net_out_num;
	for (i = 0; i < port_num; i++, prt_idx++) {
		if (prt_idx >= MAX_IO_NUM) {
			printf("Net have too much output: %u >= %u\n", prt_idx, MAX_IO_NUM);
			rval = -1;
			break;
		}
		if (port[i]->io_desc.port_slice_total_num <= 1) {
			if (strlen(port[i]->io_desc.port_demangled_name) > 0) {
				pnet->net_prt_out[prt_idx].name = port[i]->io_desc.port_demangled_name;
			} else {
				pnet->net_prt_out[prt_idx].name = port[i]->io_desc.port_name;
			}
			pnet->net_prt_out[prt_idx].sub_port[pnet->net_prt_out[prt_idx].sub_port_num] = port[i];
			pnet->net_prt_out[prt_idx].sub_port_num++;
		} else {
			if (port[i]->io_desc.port_slice_seq != 0) {
				printf("the first slice_seq should be zero, but it is %d\n",
					port[i]->io_desc.port_slice_seq);
				rval = -1;
				break;
			}
			found_num = 1;
			pnet->net_prt_out[prt_idx].name = port[i]->io_desc.port_slice_parent_name;
			pnet->net_prt_out[prt_idx].sub_port[pnet->net_prt_out[prt_idx].sub_port_num] = port[i];
			pnet->net_prt_out[prt_idx].sub_port_num++;
			slice_total_num = port[i]->io_desc.port_slice_total_num;
			while ((found_num < slice_total_num) && (i < port_num)) {
				i++;
				if (!strcmp(port[i]->io_desc.port_slice_parent_name,
					pnet->net_prt_out[prt_idx].name)) {
					//printf("output [%s] [%s]\n", port[i]->io_desc.port_name,
					//	port[i]->io_desc.port_slice_parent_name);
					if (port[i]->port_type != PORT_TYPE_OUTPUT) {
						printf("port_type should be output, but it is %d\n",
							port[i]->port_type);
						rval = -1;
						break;
					}
					if (port[i]->io_desc.port_slice_seq != found_num) {
						printf("slice_seq is wrong, %d, should be %d\n",
							port[i]->io_desc.port_slice_seq, k);
						rval = -1;
						break;
					}
					pnet->net_prt_out[prt_idx].sub_port[pnet->net_prt_out[prt_idx].sub_port_num] = port[i];
					pnet->net_prt_out[prt_idx].sub_port_num++;
					found_num++;
				}
			}
			if (rval < 0) {
				break;
			}
		}
	}
	pnet->net_prt_out_num = prt_idx;

	return rval;
}

static void gen_net_parent_port_dim(parent_port_desc_t *prt_port)
{
	port_desc_t **port = NULL;
	uint32_t i = 0, sub_port_num = 0;

	port = prt_port->sub_port;
	sub_port_num = prt_port->sub_port_num;

	if (port[0]->io_desc.port_dim_p == 1) {
		if (port[0]->io_desc.port_dim_d == 1) {
			if (port[0]->io_desc.port_dim_h == 1) {
				for (i = 0; i < sub_port_num; i++) {
					prt_port->dim.width += port[i]->io_desc.port_dim_w;
				}
				prt_port->dim.plane = port[0]->io_desc.port_dim_p;
				prt_port->dim.depth = port[0]->io_desc.port_dim_d;
				prt_port->dim.height = port[0]->io_desc.port_dim_h;
			} else {
				for (i = 0; i < sub_port_num; i++) {
					prt_port->dim.height += port[i]->io_desc.port_dim_h;
				}
				prt_port->dim.plane = port[0]->io_desc.port_dim_p;
				prt_port->dim.depth = port[0]->io_desc.port_dim_d;
				prt_port->dim.width = port[0]->io_desc.port_dim_w;
			}
		} else {
			for (i = 0; i < sub_port_num; i++) {
				prt_port->dim.depth += port[i]->io_desc.port_dim_d;
			}
			prt_port->dim.plane = port[0]->io_desc.port_dim_p;
			prt_port->dim.height = port[0]->io_desc.port_dim_h;
			prt_port->dim.width = port[0]->io_desc.port_dim_w;
		}
	} else {
		for (i = 0; i < sub_port_num; i++) {
			prt_port->dim.plane += port[i]->io_desc.port_dim_p;
		}
		prt_port->dim.depth = port[0]->io_desc.port_dim_d;
		prt_port->dim.height = port[0]->io_desc.port_dim_h;
		prt_port->dim.width = port[0]->io_desc.port_dim_w;
	}
}

static int gen_net_parent_port_size(parent_port_desc_t *prt_port)
{
	port_desc_t **port = NULL;
	uint32_t j = 0, sub_port_num = 0, first_dvi_id = 0;
	int rval = 0;

	port = prt_port->sub_port;
	sub_port_num = prt_port->sub_port_num;
	if (sub_port_num < 1) {
		printf("Abnormal sub_port_num: %u on name: %s\n",
			sub_port_num, prt_port->name);
		return -1;
	}

	if (sub_port_num > 1) {
		first_dvi_id = port[0]->dvi_id;
		prt_port->dvi_id = port[0]->dvi_id;
		for (j = 0; j < sub_port_num; j++) {
			if (first_dvi_id != port[j]->dvi_id) {
				printf("Warning: all sub port dvi_id is not same, %u != %u\n",
					first_dvi_id, port[j]->dvi_id);
			}
			prt_port->port_size += port[j]->io_desc.port_size;
			prt_port->port_dram_size += port[j]->port_dram_size;
		}
		gen_net_parent_port_dim(prt_port);
	} else if (sub_port_num == 1) {
		prt_port->dvi_id = port[0]->dvi_id;
		prt_port->port_size = port[0]->io_desc.port_size;
		prt_port->port_dram_size = port[0]->port_dram_size;

		prt_port->dim.plane = port[0]->io_desc.port_dim_p;
		prt_port->dim.depth = port[0]->io_desc.port_dim_d;
		prt_port->dim.height = port[0]->io_desc.port_dim_h;
		prt_port->dim.width = port[0]->io_desc.port_dim_w;
	}
	/* common case */
	prt_port->data_fmt.sign = port[0]->io_desc.port_data_sign;
	prt_port->data_fmt.size = port[0]->io_desc.port_data_size;
	prt_port->data_fmt.expoffset = port[0]->io_desc.port_data_expoffset;
	prt_port->data_fmt.expbits = port[0]->io_desc.port_data_expbits;

	prt_port->update_pitch = port[0]->io_desc.port_pitch;
	prt_port->dim.pitch = port[0]->io_desc.port_pitch;
	prt_port->dim.pitch_byte_offset = port[0]->io_desc.port_pitch_offset;
	prt_port->dim.pitch_bsize = port[0]->io_desc.port_pitch_bsize;
	prt_port->dim.dram_fmt = port[0]->io_desc.port_dram_format;
	prt_port->dim.bitvector = port[0]->io_desc.port_dim_bitvector;

	prt_port->port_remain_size = prt_port->port_dram_size;

	return rval;
}

void gen_net_parent_port_addr(struct net_desc *pnet)
{
	parent_port_desc_t *prt_port = NULL;
	uint32_t i = 0, num = 0;

	num = pnet->net_prt_in_num;
	prt_port = pnet->net_prt_in;
	for (i = 0; i < num; i++, prt_port++) {
		prt_port->port_dram_addr = prt_port->sub_port[0]->port_dram_addr;
	}

	num = pnet->net_prt_out_num;
	prt_port = pnet->net_prt_out;
	for (i = 0; i < num; i++, prt_port++) {
		prt_port->port_dram_addr = prt_port->sub_port[0]->port_dram_addr;
	}
}

static int gen_net_parent_port_cfg(struct net_desc *pnet)
{
	parent_port_desc_t *prt_port = NULL;
	uint32_t i = 0, num = 0;
	int rval = 0;

	num = pnet->net_prt_in_num;
	prt_port = pnet->net_prt_in;
	for (i = 0; i < num; i++, prt_port++) {
		if (gen_net_parent_port_size(prt_port) < 0) {
			printf("gen_net_parent_port_size err\n");
			rval = -1;
			break;
		}
	}

	num = pnet->net_prt_out_num;
	prt_port = pnet->net_prt_out;
	for (i = 0; i < num; i++, prt_port++) {
		if (gen_net_parent_port_size(prt_port) < 0) {
			printf("gen_net_parent_port_size err\n");
			rval = -1;
			break;
		}
	}

	return rval;
}

void show_net_parent_port(struct net_desc *pnet)
{
	parent_port_desc_t *prt_port = NULL;
	port_desc_t **port = NULL;
	io_descriptor_t *io = NULL;
	uint32_t i = 0, j = 0;

	prt_port = pnet->net_prt_in;
	printf("==== Network input and output ====\n");
	for (j = 0; j < pnet->net_prt_in_num; j++) {
		printf("Net Input: [%s] size: %u, sub_num: %u, dvi_id: %u,"
			" dim (%u, %u, %u, %u), data_fmt (%u, %u, %u, %u)\n",
			prt_port[j].name, prt_port[j].port_size,
			prt_port[j].sub_port_num, prt_port[j].dvi_id,
			prt_port[j].dim.plane, prt_port[j].dim.depth,
			prt_port[j].dim.height, prt_port[j].dim.width,
			prt_port[j].data_fmt.sign, prt_port[j].data_fmt.size,
			prt_port[j].data_fmt.expoffset, prt_port[j].data_fmt.expbits);
		if (prt_port->sub_port_num > 1) {
			port = prt_port->sub_port;
			for (i = 0; i < prt_port->sub_port_num; i++) {
				io = &port[i]->io_desc;
				printf("    Sub port %u : %s, size: %u, dim (%u, %u, %u, %u),"
					" data_fmt (%u, %u, %u, %u)\n",
					i, io->port_name, io->port_size,
					io->port_dim_p, io->port_dim_d, io->port_dim_h, io->port_dim_w,
					io->port_data_sign, io->port_data_size,
					io->port_data_expoffset, io->port_data_expbits);
			}
		}
	}

	prt_port = pnet->net_prt_out;
	for (j = 0; j < pnet->net_prt_out_num; j++) {
		printf("Net Output: [%s] size: %u, sub_num: %u, dvi_id: %u,"
			" dim (%u, %u, %u, %u), data_fmt (%u, %u, %u, %u)\n",
			prt_port[j].name, prt_port[j].port_size,
			prt_port[j].sub_port_num, prt_port[j].dvi_id,
			prt_port[j].dim.plane, prt_port[j].dim.depth,
			prt_port[j].dim.height, prt_port[j].dim.width,
			prt_port[j].data_fmt.sign, prt_port[j].data_fmt.size,
			prt_port[j].data_fmt.expoffset, prt_port[j].data_fmt.expbits);
		if (prt_port->sub_port_num > 1) {
			port = prt_port->sub_port;
			for (i = 0; i < prt_port->sub_port_num; i++) {
				io = &port[i]->io_desc;
				printf("    Sub port %u : %s, size: %u, dim (%u, %u, %u, %u), "
					"data_fmt (%u, %u, %u, %u)\n",
					i, io->port_name, io->port_size,
					io->port_dim_p, io->port_dim_d, io->port_dim_h, io->port_dim_w,
					io->port_data_sign, io->port_data_size,
					io->port_data_expoffset, io->port_data_expbits);
			}
		}
	}
	printf("==================================\n");
}

int gen_net_sub_parent_port(struct net_desc *pnet)
{
	int rval = 0;

	do {
		if (gen_net_sub_port(pnet) < 0) {
			printf("gen net sub port err\n");
			rval = -1;
			break;
		}
		if (gen_net_parent_port(pnet) < 0) {
			printf("gen net parent port err\n");
			rval = -1;
			break;
		}
		if (gen_net_parent_port_cfg(pnet) < 0) {
			printf("gen net parent port cfg err\n");
			rval = -1;
			break;
		}
		if (pnet->verbose) {
			show_net_parent_port(pnet);
		}
	} while(0);

	return rval;
}

void set_sub_and_parent_port_mfd(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	parent_port_desc_t *prt_port = NULL;
	uint32_t i = 0, j = 0, num = 0;

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		for (j = 0; j < node->dvi_desc.input_num; j++) {
			if (!node->in_port[j].no_mem) {
				node->in_port[j].port_dram_fd = pnet->mem_fd;
			} else {
				node->in_port[j].port_dram_fd = 0;
			}
		}
		for (j = 0; j < node->dvi_desc.output_num; j++) {
			if (!node->out_port[j].no_mem) {
				node->out_port[j].port_dram_fd = pnet->mem_fd;
			} else {
				node->out_port[j].port_dram_fd = 0;
			}
		}
	}

	num = pnet->net_prt_in_num;
	prt_port = pnet->net_prt_in;
	for (i = 0; i < num; i++, prt_port++) {
		prt_port->port_dram_fd = prt_port->sub_port[0]->port_dram_fd;
	}
	num = pnet->net_prt_out_num;
	prt_port = pnet->net_prt_out;
	for (i = 0; i < num; i++, prt_port++) {
		prt_port->port_dram_fd = prt_port->sub_port[0]->port_dram_fd;
	}
}

static int check_dvi_desc(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	uint32_t i = 0;
	int rval = 0;
	uint8_t first_dvi_ppv = 0;

	node = pnet->dvi_node_list;

	if (pnet->header.dvi_num > 1) {
		first_dvi_ppv = node->dvi_desc.dvi_ppv;
		node++;

		/* Check if below dvi ppv is the same as the first one */
		for (i = 1; i < pnet->header.dvi_num; i++) {
			if (node->dvi_desc.dvi_ppv != first_dvi_ppv) {
				printf("dvi id: %u ppv[%u] is not same as the first one [%u]\n",
					i, node->dvi_desc.dvi_ppv, first_dvi_ppv);
				rval = -1;
				break;
			}
			node++;
		}
		if ((!rval) && (first_dvi_ppv)) {
			printf("Enable ppv\n");
		}
	}

	return rval;
}

static int load_dvi_desc(struct net_desc *pnet,
	dvi_node_t *node, dvi_node_t *head_node,
	uint32_t dvi_pkg_pos)
{
	port_desc_t *port = NULL;
	uint32_t curr_dvi_desc_pos = 0, dvi_id = 0, net_loop = 0;
	uint32_t input_num = 0, output_num = 0;
	uint32_t i = 0;
	int rval = 0;

	if (pnet->net_loop_cnt > 1) {
		net_loop = pnet->net_loop_cnt;
	} else {
		net_loop = 1;
	}
	dvi_id = node->dvi_desc.dvi_id;
	input_num = node->dvi_desc.input_num;
	output_num = node->dvi_desc.output_num;
	if ((input_num + output_num > MAX_PORT_CNT)) {
		printf("dvi id: %u has too much port: %u + %u > %u\n",
			dvi_id, input_num, output_num, MAX_PORT_CNT);
		rval = -1;
		return rval;
	}

	node->in_port = (port_desc_t *)malloc(input_num * sizeof(port_desc_t));
	node->out_port = (port_desc_t *)malloc(output_num * sizeof(port_desc_t));
	if (!node->in_port || !node->out_port) {
		printf("malloc memory for dvi id: %u port err\n", dvi_id);
		rval = -1;
		return rval;
	}
	port = node->in_port;
	for (i = 0; i < input_num; i++, port++) {
		memset((void *)port + sizeof(port->io_desc), 0,
			(sizeof(port_desc_t) - sizeof(port->io_desc)));
	}
	port = node->out_port;
	for (i = 0; i < output_num; i++, port++) {
		memset((void *)port + sizeof(port->io_desc), 0,
			(sizeof(port_desc_t) - sizeof(port->io_desc)));
	}

	if (pnet->net_fp) {
		curr_dvi_desc_pos = ftell(pnet->net_fp);
	} else {
		curr_dvi_desc_pos = pnet->net_feed_virt_pos;
	}

	do {
		if (pnet->net_fp) {
			if (fseek(pnet->net_fp, dvi_pkg_pos, SEEK_SET) < 0) {
				perror("seek dvi pkg file pos");
				rval = -1;
				break;
			}
		} else {
			pnet->net_feed_virt_pos = dvi_pkg_pos;
		}

		port = node->in_port;
		for (i = 0; i < input_num; i++, port++) {
			if (pnet->net_fp) {
				if (fread(&port->io_desc, sizeof(io_descriptor_t), 1, pnet->net_fp) != 1) {
					perror("fread io_descriptor");
					rval = -1;
					break;
				}
			} else {
				memcpy(&port->io_desc, pnet->net_feed_virt + pnet->net_feed_virt_pos,
					sizeof(io_descriptor_t));
				pnet->net_feed_virt_pos += sizeof(io_descriptor_t);
			}
			port->dvi_id = dvi_id;
			port->port_type = PORT_TYPE_INPUT;
			port->port_dram_addr = 0;
			port->needed_by_dvi_cnt = 0;
			if (port->io_desc.port_size & 0x1f) {
				printf("Port [%s] size [0x%x] should be 32 byte Align\n",
					port->io_desc.port_name, port->io_desc.port_size);
				rval = -1;
				break;
			}
			port->port_dram_size = port->io_desc.port_size * net_loop;
			rval = build_port_dependency(head_node, port);
			if (rval < 0) {
				printf("build_port_dependency err, dvi_id: %u, port: %s\n",
					dvi_id, port->io_desc.port_name);
				break;
			}
		}
		if (rval < 0) {
			break;
		}

		port = node->out_port;
		for (i = 0; i < output_num; i++, port++) {
			if (pnet->net_fp) {
				if (fread(&port->io_desc, sizeof(io_descriptor_t), 1, pnet->net_fp) != 1) {
					perror("fread io_descriptor");
					rval = -1;
					break;
				}
			} else {
				memcpy(&port->io_desc, pnet->net_feed_virt + pnet->net_feed_virt_pos,
					sizeof(io_descriptor_t));
				pnet->net_feed_virt_pos += sizeof(io_descriptor_t);
			}
			port->dvi_id = dvi_id;
			port->port_type = PORT_TYPE_OUTPUT;
			port->port_dram_addr = 0;
			port->needed_by_dvi_cnt = 0;
			if (port->io_desc.port_size & 0x1f) {
				printf("Port [%s] size [0x%x] should be 32 byte Align\n",
					port->io_desc.port_name, port->io_desc.port_size);
				rval = -1;
				break;
			}
			port->port_dram_size = port->io_desc.port_size * net_loop;
		}
		if (rval < 0) {
			break;
		}

		if (allocate_dvi_mem(pnet, node) < 0) {
			rval = -1;
			break;
		}

		if (pnet->net_fp) {
			if (fseek(pnet->net_fp, curr_dvi_desc_pos, SEEK_SET) < 0) {
				perror("seek dvi desc offset");
				rval = -1;
				break;
			}
		} else {
			pnet->net_feed_virt_pos = curr_dvi_desc_pos;
		}
	} while (0);

	if (rval < 0) {
		if (node->in_port) {
			free(node->in_port);
			node->in_port = NULL;
		}
		if (node->out_port) {
			free(node->out_port);
			node->out_port = NULL;
		}
	}

	return rval;
}

/* Init net->dvi_node_list */
int parse_dvi_desc(struct net_desc *pnet)
{
	dvi_node_t *dvi_node_list = NULL;
	dvi_node_t *node = NULL;
	uint32_t dvi_desc_pos = 0;
	uint32_t dvi_pkg_pos = 0;
	uint32_t curr_dvi_pkg_pos = 0;
	uint32_t i = 0;
	int rval = 0;

	do {
		if (pnet->net_fp) {
			dvi_desc_pos = ftell(pnet->net_fp);
		} else  {
			dvi_desc_pos = pnet->net_feed_virt_pos;
		}
		dvi_pkg_pos = dvi_desc_pos + pnet->header.dvi_num * sizeof(dvi_desc_t);
		dvi_node_list = (dvi_node_t *)malloc(pnet->header.dvi_num * sizeof(dvi_node_t));
		if (!dvi_node_list) {
			printf("malloc dvi_node_list failed!\n");
			rval = -1;
			break;
		}
		node = dvi_node_list;
		for (i = 0; i < pnet->header.dvi_num; i++, node++) {
			memset((void *)node + sizeof(dvi_desc_t), 0,
				sizeof(dvi_node_t) - sizeof(dvi_desc_t));
		}

		node = dvi_node_list;
		curr_dvi_pkg_pos = dvi_pkg_pos;

		for (i = 0; i < pnet->header.dvi_num; i++, node++) {
			if (pnet->verbose) {
				printf("Parsing dvi id: %u desc\n", i);
			}

			/* fp stays at the start of each desc here */
			if (get_dvi_desc(pnet, &node->dvi_desc) < 0) {
				printf("get dvi id: %u desc err\n", i);
				rval = -1;
				break;
			}

			if (load_dvi_desc(pnet, node,
				dvi_node_list, curr_dvi_pkg_pos) < 0) {
				printf("load dvi id: %u desc err\n", i);
				rval = -1;
				break;
			}
			curr_dvi_pkg_pos += node->dvi_desc.dvi_pkg_size;
		}
	} while (0);

	if (dvi_node_list != NULL) {
		if (rval < 0) {
			free(dvi_node_list);
			dvi_node_list = NULL;
		} else {
			pnet->dvi_node_list = dvi_node_list;
		}
	}

	if (!rval) {
		if (check_dvi_desc(pnet) < 0) {
			printf("check dvi desc err\n");
			rval = -1;
		}
	}

	return rval;
}

int load_dvi_image_bin(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	int i = 0, ret = 0, rval = 0;

	node = pnet->dvi_node_list;

	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		if (pnet->net_fp) {
			ret = fseek(pnet->net_fp, node->dvi_file_pos, SEEK_SET);
			if (ret < 0) {
				perror("fseek dvi");
				rval = -1;
				break;
			}

			ret = fread(pnet->virt_addr + node->dvi_dram_addr,
				node->dvi_desc.dvi_img_size, 1, pnet->net_fp);
			if (ret != 1) {
				perror("fread dvi image");
				rval = -1;
				break;
			}
		} else {
			memcpy(pnet->virt_addr + node->dvi_dram_addr,
				pnet->net_feed_virt + node->dvi_file_pos,
				node->dvi_desc.dvi_img_size);
			pnet->net_feed_virt_pos = node->dvi_file_pos;
		}
	}

	return rval;
}

void update_layer_name(struct net_desc *pnet)
{
	dvi_node_t *node = NULL;
	port_desc_t *port = NULL;
	uint32_t i = 0, j = 0;

	node = pnet->dvi_node_list;

	for (i = 0; i < pnet->header.dvi_num; i++) {
		for (j = 0; j < node->dvi_desc.input_num; j++) {
			port = &node->in_port[j];
			if (port->port_type == PORT_TYPE_INPUT) {
				if (strlen(port->io_desc.port_demangled_name) > 0) {
					strncpy(port->io_desc.port_name,
						port->io_desc.port_demangled_name,
						sizeof(port->io_desc.port_name));
				}
			}
		}

		for (j = 0; j < node->dvi_desc.output_num; j++) {
			port = &node->out_port[j];
			if (port->port_type == PORT_TYPE_OUTPUT) {
				if (strlen(port->io_desc.port_demangled_name) > 0) {
					strncpy(port->io_desc.port_name,
						port->io_desc.port_demangled_name,
						sizeof(port->io_desc.port_name));
				}
			}
		}
		node++;
	}
}

void split_parent_cfg_to_sub(parent_port_desc_t *prt_port)
{
	uint32_t i = 0;

	for (i = 0; i < prt_port->sub_port_num; i++) {
		prt_port->sub_port[i]->no_mem = prt_port->no_mem;
	}
}

int split_parent_rt_flip_to_sub(parent_port_desc_t *prt_port, uint32_t is_output)
{
	uint32_t i = 0;
	int rval = 0;
	uint8_t bitmap = 0;

	for (i = 0; i < prt_port->sub_port_num; i++) {
		if (prt_port->sub_port[i]->io_desc.port_drotate_bit_offset) {
			/* prevent to set rotate, hflip on output */
			if (is_output) {
				bitmap = prt_port->rotate_flip_bitmap;
				if ((bitmap & (1 << DROTATE_BIT)) >> DROTATE_BIT) {
					printf("Not support rotate on output [%s], 0x%x\n",
						prt_port->name, bitmap);
					rval = -1;
					break;
				}
				if ((bitmap & (1 << HFLIP_BIT)) >> HFLIP_BIT) {
					printf("Not support hflip on output [%s], 0x%x\n",
						prt_port->name, bitmap);
					rval = -1;
					break;
				}
			}

			prt_port->sub_port[i]->rotate_flip_bitmap =
				prt_port->rotate_flip_bitmap;

			if (prt_port->sub_port[i]->rotate_flip_in_dvi !=
				prt_port->sub_port[i]->rotate_flip_bitmap) {
				printf("Set port [%s], sub [%s], rotate-flip: 0x%x\n",
					prt_port->name, prt_port->sub_port[i]->io_desc.port_name,
					prt_port->sub_port[i]->rotate_flip_bitmap);
			}
		}
	}

	return rval;
}

void split_parent_pitch_to_sub(parent_port_desc_t *prt_port)
{
	uint32_t i = 0;

	for (i = 0; i < prt_port->sub_port_num; i++) {
		prt_port->sub_port[i]->update_pitch = prt_port->update_pitch;

		if (prt_port->sub_port[i]->io_desc.port_pitch !=
			prt_port->sub_port[i]->update_pitch) {
			printf("Set port [%s], sub [%s], update_pitch: %u\n",
				prt_port->name, prt_port->sub_port[i]->io_desc.port_name,
				prt_port->sub_port[i]->update_pitch);
		}
	}
}

int split_parent_addr_to_sub(parent_port_desc_t *prt_port)
{
	uint32_t addr = 0, i = 0;
	int rval = 0;

	addr = prt_port->port_dram_addr;
	for (i = 0; i < prt_port->sub_port_num; i++) {
		prt_port->sub_port[i]->port_dram_addr = addr;
		addr += prt_port->sub_port[i]->port_dram_size;
		prt_port->sub_port[i]->port_dram_fd = prt_port->port_dram_fd;
	}

	if (addr != (prt_port->port_dram_addr + prt_port->port_dram_size)) {
		printf("Abnormal parent address addr: 0x%x != 0x%x + 0x%x\n",
			addr, prt_port->port_dram_addr, prt_port->port_dram_size);
		rval = -1;
	}

	return rval;
}

int get_port_cfg(struct net_desc *pnet,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	parent_port_desc_t *prt_port = NULL;
	uint32_t i = 0, k = 0;
	uint32_t found = 0;
	int rval = 0;

	do {
		if (net_in) {
			prt_port = pnet->net_prt_in;
			for (i = 0; i < pnet->net_prt_in_num; i++, prt_port++) {
				found = 0;
				for (k = 0; k < net_in->in_num; k++) {
					if (strcmp(prt_port->name, net_in->in_desc[k].name) == 0) {
						found = 1;
						prt_port->no_mem = net_in->in_desc[k].no_mem;
						prt_port->rotate_flip_bitmap = net_in->in_desc[k].rotate_flip_bitmap;
						net_in->in_desc[k].size = prt_port->port_size;
						net_in->in_desc[k].dim = prt_port->dim;
						net_in->in_desc[k].data_fmt = prt_port->data_fmt;

						split_parent_cfg_to_sub(prt_port);
						rval = split_parent_rt_flip_to_sub(prt_port, 0);
					}
				}
				if (rval < 0) {
					break;
				}
				if (!found) {
					printf("Please specify input [%s] for init\n", prt_port->name);
					rval = -1;
					break;
				}
			}
		}

		if (net_out) {
			for (k = 0; k < net_out->out_num; k++) {
				prt_port = pnet->net_prt_out;
				found = 0;
				for (i = 0; i < pnet->net_prt_out_num; i++, prt_port++) {
					if (strcmp(prt_port->name, net_out->out_desc[k].name) == 0) {
						found = 1;
						prt_port->no_mem = net_out->out_desc[k].no_mem;
						prt_port->rotate_flip_bitmap = net_out->out_desc[k].rotate_flip_bitmap;
						net_out->out_desc[k].size = prt_port->port_size;
						net_out->out_desc[k].dim = prt_port->dim;
						net_out->out_desc[k].data_fmt = prt_port->data_fmt;

						split_parent_cfg_to_sub(prt_port);
						rval = split_parent_rt_flip_to_sub(prt_port, 1);
					}
				}
				if (rval < 0) {
					break;
				}
				if (!found) {
					printf("Not found output [%s] for init\n", net_out->out_desc[k].name);
					rval = -1;
					break;
				}
			}
		}
	} while (0);

	if (net_in || net_out) {
		if (!found) {
			show_net_parent_port(pnet);
		}
	}

	return rval;
}

int set_port_cfg(struct net_desc *pnet,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	parent_port_desc_t *prt_port = NULL;
	struct input_desc *in = NULL;
	struct output_desc *out = NULL;
	uint32_t i = 0, k = 0;
	uint32_t found = 0, verbose = 0;
	uint32_t align_size = 0;
	int rval = 0;

	verbose = pnet->verbose;
	do {
		/* input */
		if (net_in) {
		prt_port = pnet->net_prt_in;
		for (i = 0; i < pnet->net_prt_in_num; i++, prt_port++) {
			found = 0;
			for (k = 0; k < net_in->in_num; k++) {
				in = &net_in->in_desc[k];
				if (strcmp(prt_port->name, in->name) == 0) {
					found = 1;
					if (prt_port->no_mem == 0) {
						in->virt = pnet->virt_addr + prt_port->port_dram_addr;
						in->size = prt_port->port_size;
						in->addr = pnet->phy_addr + prt_port->port_dram_addr;
					} else {
						if (in->size != prt_port->port_size) {
							printf("input [%s] size mismatch, set %u, but dvi is %u\n",
								in->name, in->size, prt_port->port_size);
							rval = -1;
							break;
						}
						if (in->addr) {
							prt_port->port_dram_addr = in->addr - pnet->phy_addr;
							rval = split_parent_addr_to_sub(prt_port);
							if (verbose) {
								printf("Set input [%s], addr: 0x%x in load\n",
									in->name, in->addr);
							}
						} else {
							printf("Invalid input [%s] addr: 0x%x in load\n",
								in->name, in->addr);
							rval = -1;
							break;
						}
					}
					if (in->update_pitch) {
						align_size = ROUND_UP(in->update_pitch, ALIGN_32);
						if (in->update_pitch != align_size) {
							printf("input update_pitch is not 32 byte align: %u\n",
								in->update_pitch);
							rval = -1;
							break;
						}
						if (in->update_pitch < (prt_port->dim.width << prt_port->data_fmt.size)) {
							printf("input update_pitch is less than width size: %u < %u\n",
								in->update_pitch, (prt_port->dim.width << prt_port->data_fmt.size));
							rval = -1;
							break;
						}
						prt_port->update_pitch = in->update_pitch;
						split_parent_pitch_to_sub(prt_port);
					}
					in->dim = prt_port->dim;
					in->data_fmt = prt_port->data_fmt;
				}
			}
			if (rval < 0) {
				break;
			}
			if (!found) {
				printf("Please secify input [%s] in load\n", prt_port->name);
				rval = -1;
				break;
			}
		}
		}

		/* output */
		if (net_out) {
		for (k = 0; k < net_out->out_num; k++) {
			out = &net_out->out_desc[k];
			prt_port = pnet->net_prt_out;
			found = 0;
			for (i = 0; i < pnet->net_prt_out_num; i++, prt_port++) {
				if (strcmp(prt_port->name, out->name) == 0) {
					found = 1;
					if (prt_port->no_mem == 0) {
						out->virt = pnet->virt_addr + prt_port->port_dram_addr;
						out->size = prt_port->port_size;
						out->addr = pnet->phy_addr + prt_port->port_dram_addr;
					} else {
						if (out->size != prt_port->port_size) {
							printf("output [%s] size mismatch, set %u, but dvi is %u\n",
								out->name, out->size, prt_port->port_size);
							rval = -1;
							break;
						}
						if (out->addr) {
							prt_port->port_dram_addr = out->addr - pnet->phy_addr;
							rval = split_parent_addr_to_sub(prt_port);
							if (rval < 0) {
								break;
							}
							if (verbose) {
								printf("Set output [%s], addr: 0x%x in load\n",
									out->name, out->addr);
							}
							/* Need update itm_frm_out port addr if set output addr */
							rval = allocate_itm_from_out_mem(pnet);
						} else {
							printf("Invalid output [%s] addr: 0x%x in load\n",
								out->name, out->addr);
							rval = -1;
							break;
						}
					}
					if (out->update_pitch) {
						align_size = ROUND_UP(out->update_pitch, ALIGN_32);
						if (out->update_pitch != align_size) {
							printf("output update_pitch is not 32 byte align: %u\n",
								out->update_pitch);
							rval = -1;
							break;
						}
						if (out->update_pitch < (prt_port->dim.width << prt_port->data_fmt.size)) {
							printf("output update_pitch is less than width size: %u < %u\n",
								out->update_pitch, (prt_port->dim.width << prt_port->data_fmt.size));
							rval = -1;
							break;
						}
						prt_port->update_pitch = out->update_pitch;
						split_parent_pitch_to_sub(prt_port);
					}
					out->dim = prt_port->dim;
					out->data_fmt = prt_port->data_fmt;
				}
			}
			if (rval < 0) {
				break;
			}
			if (!found) {
				printf("Not found output [%s] in load\n", out->name);
				rval = -1;
				break;
			}
		}
		}
	} while (0);

	return rval;
}

int update_port_cfg(struct net_desc *pnet,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out,
	uint32_t *need_update_port, uint32_t *need_update_poke)
{
	struct input_desc *in = NULL;
	struct output_desc *out = NULL;
	parent_port_desc_t *prt_port = NULL;
	uint32_t i = 0, k = 0;
	uint32_t found = 0, verbose = 0;
	uint32_t align_size = 0;
	int rval = 0;

	if (net_in == NULL && net_out == NULL) {
		return 0;
	}

	verbose = pnet->verbose;
	do {
		if (net_in) {
			for (k = 0; k < net_in->in_num; k++) {
				in = &net_in->in_desc[k];
				prt_port = pnet->net_prt_in;
				found = 0;
				for (i = 0; i < pnet->net_prt_in_num; i++, prt_port++) {
					if (strcmp(prt_port->name, in->name) == 0) {
						found = 1;
						if (prt_port->no_mem) {
							if (!in->addr) {
								printf("Invalid input [%s] addr: 0x%x in run\n",
									in->name, in->addr);
								rval = -1;
								break;
							}
							if ((in->addr - pnet->phy_addr) != prt_port->port_dram_addr) {
								*need_update_port = 1;
								prt_port->port_dram_addr = in->addr - pnet->phy_addr;
								rval = split_parent_addr_to_sub(prt_port);
								if (rval < 0) {
									break;
								}
								if (verbose) {
									printf("Update input [%s] addr: 0x%x\n",
										in->name, in->addr);
								}
							}
						}

						if (in->rotate_flip_bitmap != prt_port->rotate_flip_bitmap) {
							*need_update_poke = 1;
							prt_port->rotate_flip_bitmap = in->rotate_flip_bitmap;
							rval = split_parent_rt_flip_to_sub(prt_port, 0);
						}
						if ((in->update_pitch) &&
							(in->update_pitch != prt_port->update_pitch)) {
							align_size = ROUND_UP(in->update_pitch, ALIGN_32);
							if (in->update_pitch != align_size) {
								printf("input update_pitch is not 32 byte align: %u\n",
									in->update_pitch);
								rval = -1;
								break;
							}
							if (in->update_pitch < (prt_port->dim.width << prt_port->data_fmt.size)) {
								printf("input update_pitch is less than width size: %u < %u\n",
									in->update_pitch, (prt_port->dim.width << prt_port->data_fmt.size));
								rval = -1;
								break;
							}

							*need_update_poke = 1;
							prt_port->update_pitch = in->update_pitch;
							split_parent_pitch_to_sub(prt_port);
						}
						break;
					}
				}
				if (!found) {
					printf("Not found input [%s] in run\n", in->name);
					rval = -1;
				}
				if (rval < 0) {
					break;
				}
			}
		}
		if (rval < 0) {
			break;
		}

		if (net_out) {
			for (k = 0; k < net_out->out_num; k++) {
				out = &net_out->out_desc[k];
				prt_port = pnet->net_prt_out;
				found = 0;
				for (i = 0; i < pnet->net_prt_out_num; i++, prt_port++) {
					if (strcmp(prt_port->name, out->name) == 0) {
						found = 1;
						if (prt_port->no_mem) {
							if (!out->addr) {
								printf("Invalid output [%s] addr: 0x%x in run\n",
									out->name, out->addr);
								rval = -1;
								break;
							}
							if ((out->addr - pnet->phy_addr) != prt_port->port_dram_addr) {
								*need_update_port = 1;
								prt_port->port_dram_addr = out->addr - pnet->phy_addr;
								rval = split_parent_addr_to_sub(prt_port);
								if (rval < 0) {
									break;
								}
								/* Need update itm_frm_out port addr if set output addr */
								rval = allocate_itm_from_out_mem(pnet);
								if (rval < 0) {
									break;
								}
								if (verbose) {
									printf("Update output [%s] addr: 0x%x\n",
										out->name, out->addr);
								}
							}
						}
						if (out->rotate_flip_bitmap != prt_port->rotate_flip_bitmap) {
							*need_update_poke = 1;
							prt_port->rotate_flip_bitmap = out->rotate_flip_bitmap;
							rval = split_parent_rt_flip_to_sub(prt_port, 1);
						}
						if ((out->update_pitch) &&
							(out->update_pitch != prt_port->update_pitch)) {
							align_size = ROUND_UP(out->update_pitch, ALIGN_32);
							if (out->update_pitch != align_size) {
								printf("output update_pitch is not 32 byte align: %u\n",
									out->update_pitch);
								rval = -1;
								break;
							}
							if (out->update_pitch < (prt_port->dim.width << prt_port->data_fmt.size)) {
								printf("output update_pitch is less than width size: %u < %u\n",
									out->update_pitch, (prt_port->dim.width << prt_port->data_fmt.size));
								rval = -1;
								break;
							}

							*need_update_poke = 1;
							prt_port->update_pitch = out->update_pitch;
							split_parent_pitch_to_sub(prt_port);
						}
						break;
					}
				}
				if (!found) {
					printf("Not found output [%s] in run\n", out->name);
					rval = -1;
				}
				if (rval < 0) {
					break;
				}
			}
		}
		if (rval < 0) {
			break;
		}
	} while (0);

	return rval;
}

