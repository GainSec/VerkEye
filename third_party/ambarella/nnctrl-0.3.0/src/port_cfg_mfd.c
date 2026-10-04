/*******************************************************************************
 * port_cfg_mfd.c
 *
 * History:
 *    2020/05/15  - [Tao Wu] created
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
#include "parser.h"
#include "nnctrl_priv.h"
#include "utils.h"

int check_io_num_mfd(struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out)
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

int get_port_cfg_mfd(struct net_desc *pnet,
	struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out)
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

int set_port_cfg_mfd(struct net_desc *pnet,
	struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out)
{
	parent_port_desc_t *prt_port = NULL;
	struct input_mfd_desc *in = NULL;
	struct output_mfd_desc *out = NULL;
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
						in->mem_fd = prt_port->port_dram_fd;
						in->fd_offset = prt_port->port_dram_addr;
					} else {
						if (in->size != prt_port->port_size) {
							printf("input [%s] size mismatch, set %lu, but dvi is %u\n",
								in->name, in->size, prt_port->port_size);
							rval = -1;
							break;
						}
						prt_port->port_dram_addr = in->fd_offset;
						prt_port->port_dram_fd = in->mem_fd;
						rval = split_parent_addr_to_sub(prt_port);
						if (verbose) {
							printf("Set input [%s], addr fd: %d, offset: 0x%lx in load\n",
								in->name, in->mem_fd, in->fd_offset);
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
						out->mem_fd = prt_port->port_dram_fd;
						out->fd_offset = prt_port->port_dram_addr;
					} else {
						if (out->size != prt_port->port_size) {
							printf("output [%s] size mismatch, set %lu, but dvi is %u\n",
								out->name, out->size, prt_port->port_size);
							rval = -1;
							break;
						}
						prt_port->port_dram_addr = out->fd_offset;
						prt_port->port_dram_fd = out->mem_fd;
						rval = split_parent_addr_to_sub(prt_port);
						if (verbose) {
							printf("Set output [%s], addr fd: %d, offset: 0x%lx in load\n",
								out->name, out->mem_fd, out->fd_offset);
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

int update_port_cfg_mfd(struct net_desc *pnet,
	struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out,
	uint32_t *need_update_port, uint32_t *need_update_poke)
{
	struct input_mfd_desc *in = NULL;
	struct output_mfd_desc *out = NULL;
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
							if ((prt_port->port_dram_fd != in->mem_fd) ||
								(prt_port->port_dram_addr != in->fd_offset)) {
								*need_update_port = 1;
								prt_port->port_dram_fd = in->mem_fd;
								prt_port->port_dram_addr = in->fd_offset;
								rval = split_parent_addr_to_sub(prt_port);
								if (rval < 0) {
									break;
								}
								if (verbose) {
									printf("Update input [%s] fd: %d, offset: 0x%lx\n",
										in->name, in->mem_fd, in->fd_offset);
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
							if ((prt_port->port_dram_fd != out->mem_fd) ||
								(prt_port->port_dram_addr != out->fd_offset)) {
								*need_update_port = 1;
								prt_port->port_dram_fd = out->mem_fd;
								prt_port->port_dram_addr = out->fd_offset;
								rval = split_parent_addr_to_sub(prt_port);
								if (rval < 0) {
									break;
								}
								if (verbose) {
									printf("Update output [%s] fd: %d, offset: 0x%lx\n",
										out->name, out->mem_fd, out->fd_offset);
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

